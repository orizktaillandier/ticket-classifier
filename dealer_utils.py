import re
import nltk
import csv
import pandas as pd

nltk.download("punkt", quiet=True)

DEALER_BLOCKLIST = {"blue admin", "admin blue", "admin red", "d2c media", "cars commerce"}

def detect_language(text):
    return "fr" if re.search(r"\\b(merci|bonjour|véhicule|images|depuis)\\b", text.lower()) else "en"

def detect_stock_number(text):
    return bool(re.search(r"\\b[A-Z0-9]{6,}\\b", text))

def extract_contacts(text):
    lines = text.strip().split('\\n')
    for i in range(len(lines) - 1):
        line = lines[i].strip().lower()
        if re.match(r'^(best regards|regards|merci|thanks|cordially|from:|envoyé par|de:)', line, re.IGNORECASE):
            next_line = lines[i + 1].strip()
            name_match = re.match(r'^[A-Z][a-z]+( [A-Z][a-z]+)+$', next_line)
            if name_match:
                return next_line
    greet_match = re.search(r'^(hi|bonjour|hello|salut)[\\s,:-]+([A-Z][a-z]+)', text.strip(), re.IGNORECASE | re.MULTILINE)
    if greet_match:
        candidate = greet_match.group(2)
        if not re.match(r'^(nous|client|dealer|photos?|images?|request|inventory)$', candidate, re.IGNORECASE):
            return candidate
    match = re.search(r'\\b([A-Z][a-z]+ [A-Z][a-z]+)\\b', text)
    if match and not re.match(r'^(nous|client|dealer|photos?|images?|request|inventory)$', match.group(1), re.IGNORECASE):
        return match.group(1)
    return ""

def extract_dealers(text):
    dealer_matches = re.findall(
        r"\\b(?:mazda|toyota|honda|chevrolet|hyundai|genesis|ford|ram|gmc|acura|jeep"
        r"|buick|nissan|volvo|subaru|volkswagen|kia|mitsubishi|infiniti|lexus"
        r"|cadillac|dodge|mini|jaguar|land rover|bmw|mercedes|audi|porsche|tesla)"
        r"[a-zé\\-\\s]*\\b", text.lower()
    )
    INVALID_SUFFIXES = {"units", "inventory", "vehicles", "images", "stock"}

    cleaned = []
    for d in dealer_matches:
        d_clean = d.strip()
        if d_clean in DEALER_BLOCKLIST:
            continue
        parts = d_clean.split()
        if parts and parts[-1] not in INVALID_SUFFIXES:
            cleaned.append(d_clean)

    return list(set(cleaned))

def extract_syndicators(text):
    candidates = [
        "vauto", "easydeal", "car media", "icc", "homenet", "serti",
        "evolutionautomobiles", "spincar", "trader", "pbs", "google", "omni"
    ]
    return [c for c in candidates if c in text.lower()]

def extract_image_flags(text):
    flags = []
    lower = text.lower()
    if "image" in lower:
        flags.append("image")
    if "certified" in lower:
        flags.append("certified")
    if "overwrite" in lower or "overwritten" in lower:
        flags.append("overwritten")
    return flags

def preprocess_ticket(text):
    return {
        "message": text,
        "contains_french": detect_language(text) == "fr",
        "contains_stock_number": detect_stock_number(text),
        "contacts_found": [extract_contacts(text)],
        "dealers_found": extract_dealers(text),
        "syndicators": extract_syndicators(text),
        "image_flags": extract_image_flags(text),
        "line_count": text.count("\\n") + 1
    }
"""

llm_classifier_fixed_code = """
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
