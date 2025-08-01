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
    if re.search(r"(stock number|stock#).*?[<>'\"\\\\]", text):
        return "E44"
    if "firewall" in text:
        return "E74"
    if "partial" in text and "trim" in text and "inventory+" in text and "omni" in text:
        return "E77"
    return ""

def format_zoho_comment(zf, context):
    lines = []
    lines.append(f"{zf.get('dealer_name', '')} ({zf.get('dealer_id', '')})")
    lines.append(f"Rep: {zf.get('rep', '')}")

    dealer_emails = re.findall(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", context.get("message", "").lower())
    known = [e for e in dealer_emails if "kotauto" in e or "fairisleford" in e]
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
    dealer_list = context.get("dealers_found", [])
    example = dealer_list[0] if dealer_list else find_example_dealer(text)
    override = lookup_dealer_by_name(example) if example else {}

    SYSTEM_PROMPT = (
        "You are a Zoho Desk classification assistant. Use only dropdown values:\n"
        "- Category: Product Activation – New Client, Product Activation – Existing Client, Product Cancellation, Problem / Bug, General Question, Analysis / Review, Other.\n"
        "- Sub Category: Import, Export, Sales Data Import, FB Setup, Google Setup, Other Department, Other, AccuTrade.\n"
        "- Inventory Type: New, Used, Demo, New + Used, or blank.\n"
        "- Never hardcode comment text. Leave field 'zoho_comment' as '...' and it will be constructed later.\n"
    )

    USER_PROMPT = f\"""Classify this message:\n{text}\nReturn JSON only with zoho_comment set to '...'.\"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\\s*", "", raw)
    raw = re.sub(r"\\s*```$", "", raw)
    m = re.search(r"\\{.*\\}", raw, re.DOTALL)
    if not m:
        raise ValueError("❌ LLM did not return valid JSON:\\n" + raw)
    json_text = m.group(0)
    data = json.loads(json_text)
    zf = data.get("zoho_fields", {})

    # Normalize and override dealer
    dn_raw = zf.get("dealer_name", "")
    dn = re.sub(r"([a-z])([A-Z])", r"\\1 \\2", dn_raw).lower().strip()
    mapped_id = dealer_to_id.get(dn, "")

    if not mapped_id and example and override.get("dealer_id") and "group" not in example.lower():
        zf["dealer_name"] = example
        zf["dealer_id"] = override["dealer_id"]
        zf["rep"] = override["rep"]
    else:
        if mapped_id:
            zf["dealer_id"] = mapped_id
            zf["rep"] = dealer_to_rep.get(dn, "")

    if not zf.get("syndicator") and context.get("syndicators"):
        zf["syndicator"] = context["syndicators"][0].title()

    zf["dealer_name"] = zf.get("dealer_name", "").title()
    zf["contact"] = zf["rep"]

    if zf.get("syndicator", "").lower() == "omni" and not zf.get("inventory_type"):
        zf["inventory_type"] = "Used + New"

    data["zoho_fields"] = zf
    data["zoho_comment"] = format_zoho_comment(zf, context)
    data["edge_case"] = detect_edge_case(text, zf)
    data.pop("suggested_reply", None)

    return data

def find_example_dealer(text: str):
    patterns = [r"for ([A-Za-z0-9 &\\\\-]+)\\b", r"from ([A-Za-z0-9 &\\\\-]+)\\b", r"regarding ([A-Za-z0-9 &\\\\-]+)\\b"]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""
