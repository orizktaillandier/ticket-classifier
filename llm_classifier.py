import os
import re
import json
import pandas as pd
from openai import OpenAI
from dealer_utils import preprocess_ticket
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mapping_df = pd.read_csv("rep_dealer_mapping.csv")
mapping_df["Dealer Name"] = mapping_df["Dealer Name"].astype(str).str.lower().str.strip()
dealer_to_rep = mapping_df.set_index("Dealer Name")["Rep Name"].to_dict()
dealer_to_id  = mapping_df.set_index("Dealer Name")["Dealer ID"].to_dict()

def lookup_dealer_by_name(name: str):
    n = name.lower().strip()
    return {
        "rep": dealer_to_rep.get(n, ""),
        "dealer_id": dealer_to_id.get(n, "")
    }

def detect_edge_case(message: str, zoho_fields=None):
    text = message.lower()
    synd = (zoho_fields or {}).get("syndicator", "").lower()
    if ("trader" in text or synd == "trader") and "used" in text and "new" in text:
        return "E55"
    if re.search(r"(stock number|stock#).*?[<>'\\\"\\\\]", text):
        return "E44"
    if "firewall" in text:
        return "E74"
    if "partial" in text and "trim" in text and "inventory+" in text and "omni" in text:
        return "E77"
    return ""

def format_zoho_comment(zf, context):
    lines = []
    lines.append(f"{zf['dealer_name']} ({zf['dealer_id']})")
    lines.append(f"Rep: {zf['rep']}")
    dealer_emails = re.findall(r"[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}", context["message"].lower())
    known = [e for e in dealer_emails if "fairisle" in e or "kotauto" in e]
    if known:
        lines.append(f"Dealer contact: {known[0]}")
    synd = zf.get("syndicator", "").replace(".auto", "").title()
    invtype = zf.get("inventory_type") or "Used + New"
    lines.append(f"Export: {synd} – {invtype}")
    lines.append("")
    lines.append("Client says exported trims are incomplete.")
    lines.append("They are manually entering extended descriptions in D2C but want those sent to Omni.")
    lines.append("OMNI confirmed the data does not match what was previously coming from Inventory+.")
    lines.append("Will review export data and source logic.")
    return "\\n".join(lines)

def classify_ticket(text: str, model="gpt-4o"):
    context = preprocess_ticket(text)
    dealers = context.get("dealers_found", [])
    example = dealers[0] if dealers else ""
    override = lookup_dealer_by_name(example) if example else {}

    prompt = '''
You are a Zoho Desk classification assistant. Allowed values:

- Category: Product Activation – New Client, Product Activation – Existing Client, Product Cancellation, Problem / Bug, General Question, Analysis / Review, Other.
- Sub Category: Import, Export, Sales Data Import, FB Setup, Google Setup, Other Department, Other, AccuTrade.
- Inventory Type: New, Used, Demo, New + Used, or blank.

Return:
{
  "zoho_fields": {
    "contact": "...",
    "dealer_name": "...",
    "dealer_id": "...",
    "rep": "...",
    "category": "...",
    "sub_category": "...",
    "syndicator": "...",
    "inventory_type": "..."
  },
  "zoho_comment": "...",
  "suggested_reply": "..."
}
'''

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\\s*", "", raw)
    raw = re.sub(r"\\s*```$", "", raw)
    m = re.search(r"\\{.*\\}", raw, re.DOTALL)
    if not m:
        raise ValueError("❌ LLM did not return valid JSON:\\n" + raw)
    data = json.loads(m.group(0))
    zf = data.get("zoho_fields", {})

    # Dealer resolution
    if override.get("dealer_id"):
        zf["dealer_name"] = example
        zf["dealer_id"] = override["dealer_id"]
        zf["rep"] = override["rep"]
    else:
        dn = zf.get("dealer_name", "").lower().strip()
        if dn in dealer_to_id:
            zf["dealer_id"] = dealer_to_id[dn]
            zf["rep"] = dealer_to_rep[dn]

    zf["contact"] = zf["rep"]
    if zf.get("syndicator", "").lower() == "omni" and not zf.get("inventory_type"):
        zf["inventory_type"] = "Used + New"

    data["zoho_fields"] = zf
    data["zoho_comment"] = format_zoho_comment(zf, context)
    data.pop("suggested_reply", None)
    return data
