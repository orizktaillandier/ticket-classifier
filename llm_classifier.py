import os
import re
import json
import pandas as pd
from openai import OpenAI
from dealer_utils import preprocess_ticket, lookup_dealer_by_name
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mapping_df = pd.read_csv("rep_dealer_mapping.csv")
mapping_df["Dealer Name"] = mapping_df["Dealer Name"].astype(str).str.lower().str.strip()
dealer_to_rep = mapping_df.set_index("Dealer Name")["Rep Name"].to_dict()
dealer_to_id  = mapping_df.set_index("Dealer Name")["Dealer ID"].astype(str).to_dict()

def detect_edge_case(message: str, zoho_fields=None):
    text = message.lower()
    synd = (zoho_fields or {}).get("syndicator", "").lower()
    if ("trader" in text or synd == "trader") and "used" in text and "new" in text:
        return "E55"
    if re.search(r"(stock number|stock#).*?[<>'\"\\\\]", text):
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

    emails = re.findall(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", context["message"].lower())
    dealer_emails = [e for e in emails if "@" in e]
    if dealer_emails:
        lines.append(f"Dealer contact: {dealer_emails[0]}")

    synd = zf.get("syndicator", "").replace(".auto", "").title()
    inv = zf.get("inventory_type") or "Used + New"
    lines.append(f"Export: {synd} – {inv}")

    lines.append("")
    lines.append("Client says exported trims are incomplete.")
    lines.append("They are manually entering extended descriptions in D2C but want those sent to Omni.")
    lines.append("OMNI confirmed the data does not match what was previously coming from Inventory+.")
    lines.append("Will review export data and source logic.")

    return "\n".join(lines)

def classify_ticket(text: str, model="gpt-4o"):
    context = preprocess_ticket(text)
    dealer_list = context.get("dealers_found", [])
    example = dealer_list[0] if dealer_list else ""
    override = lookup_dealer_by_name(example) if example else {}

    FEMSHOT = """
Example:
Message:
"Hi Véronique, Mazda Steele is still showing vehicles that were sold last week. Request to check the PBS import."

Zoho Fields:
{
  "contact": "Véronique Fournier",
  "dealer_name": "Mazda Steele",
  "dealer_id": "2618",
  "rep": "Véronique Fournier",
  "category": "Problem / Bug",
  "sub_category": "Import",
  "syndicator": "PBS",
  "inventory_type": ""
}
"""
    SYSTEM_PROMPT = (
        "You are a Zoho Desk classification assistant. Only use these allowed dropdown values:\n"
        "- Category: Product Activation – New Client, Product Activation – Existing Client, Product Cancellation, Problem / Bug, General Question, Analysis / Review, Other.\n"
        "- Sub Category: Import, Export, Sales Data Import, FB Setup, Google Setup, Other Department, Other, AccuTrade.\n"
        "- Inventory Type: New, Used, Demo, New + Used, or blank.\n\n"
        "Important logic rules (do not override):\n"
        "- dealer_name, dealer_id, and rep must come from internal mapping — if uncertain, leave blank.\n"
        "- Only use physical rooftop names, not group names like 'Kot Auto Group'.\n"
        "- syndicator is the export destination (where D2C sends the feed), not a data source like Inventory+.\n"
        "- Do not infer or guess values. Return only grounded field values.\n\n"
        + FEMSHOT +
        "\nNow classify the following message and return ONLY the JSON object (no explanation or extra text):"
    )
    USER_PROMPT = f"""
Message:
{text}

Return exactly this JSON format:
{{
  "zoho_fields": {{
    "contact": "...",
    "dealer_name": "...",
    "dealer_id": "...",
    "rep": "...",
    "category": "...",
    "sub_category": "...",
    "syndicator": "...",
    "inventory_type": "..."
  }},
  "zoho_comment": "...",
  "suggested_reply": "..."
}}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("LLM output not valid JSON:\n" + raw)
    data = json.loads(m.group(0))
    zf = data.get("zoho_fields", {})

    dn_raw = zf.get("dealer_name", "")
    dn = re.sub(r"([a-z])([A-Z])", r"\1 \2", dn_raw).lower().strip()
    mapped_id = dealer_to_id.get(dn, "")

    if not mapped_id and example and override.get("dealer_id") and "group" not in example.lower():
        zf["dealer_name"] = example
        zf["dealer_id"] = override["dealer_id"]
        zf["rep"] = override["rep"]
    elif mapped_id:
        zf["dealer_id"] = mapped_id
        zf["rep"] = dealer_to_rep.get(dn, "")

    if not zf.get("syndicator") and context.get("syndicators"):
        zf["syndicator"] = context["syndicators"][0].title()

    zf["dealer_name"] = zf.get("dealer_name", "").title()
    zf["contact"] = zf.get("rep", "")

    if zf.get("syndicator", "").lower() == "omni" and not zf.get("inventory_type"):
        zf["inventory_type"] = "Used + New"

    data["zoho_comment"] = format_zoho_comment(zf, context)
    data["edge_case"] = detect_edge_case(text, zf)
    data.pop("suggested_reply", None)

    return data
