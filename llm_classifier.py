import os
import json
import re
import pandas as pd
from openai import OpenAI
from fuzzywuzzy import process, fuzz
from dotenv import load_dotenv
from dealer_utils import preprocess_ticket
from datetime import datetime

LOGFILE = "ticket_classifier_log.jsonl"

def write_log(input_text, output, edge_case=None):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_text,
        "output": output,
        "edge_case": edge_case or ""
    }
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Load OpenAI API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load and normalize rep map
rep_mapping = pd.read_csv("./rep_dealer_mapping.csv")
rep_mapping["Dealer Name Normalized"] = (
    rep_mapping["Dealer Name"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^a-z0-9\s]", "", regex=True)
    .str.split()
    .apply(lambda x: " ".join(sorted(x)))
)
dealer_lookup = {
    name: (row["Dealer Name"], row["Rep Name"], str(row["Dealer ID"]))
    for _, row in rep_mapping.iterrows()
    for name in [row["Dealer Name Normalized"]]
}
DEALER_KEYS = list(dealer_lookup.keys())

def normalize_dealer_name(raw_name):
    return " ".join(sorted(re.sub(r"[^a-z0-9\s]", "", str(raw_name).lower()).split()))

def robust_dealer_match(candidate_names):
    for name in candidate_names:
        norm = normalize_dealer_name(name)
        if norm in dealer_lookup:
            return dealer_lookup[norm]
    for name in candidate_names:
        norm = normalize_dealer_name(name)
        best_match = process.extractOne(norm, DEALER_KEYS, scorer=fuzz.token_set_ratio)
        if best_match and best_match[1] >= 90:
            return dealer_lookup[best_match[0]]
    return ("", "", "")

def extract_best_dealer(context, raw_message):
    candidates = context.get("dealers_found", [])
    candidates = [c for c in candidates if c] + [raw_message]
    return robust_dealer_match(candidates)

def extract_syndicator(text):
    known = [
        "vauto", "easydeal", "car media", "icc", "homenet", "serti",
        "evolutionautomobiles", "spincar", "trader", "pbs"
    ]
    lower = text.lower()
    for s in known:
        if re.search(rf"\b{s}\b", lower):
            return s.title() if s != "icc" else "ICC"
    return ""

def validate_fields(fields):
    valid = {
        "category": [
            "Product Activation – New Client", "Product Activation – Existing Client",
            "Product Cancellation", "Problem / Bug", "General Question",
            "Analysis / Review", "Other"
        ],
        "sub_category": [
            "Import", "Export", "Sales Data Import", "FB Setup",
            "Google Setup", "Other Department", "Other", "AccuTrade"
        ],
        "inventory_type": ["New", "Used", "Demo", "New + Used", ""]
    }
    for k in valid:
        if fields.get(k) not in valid[k]:
            fields[k] = ""
    return fields

def detect_edge_case(message, zoho_fields=None):
    text = message.lower()
    syndicator = (zoho_fields or {}).get("syndicator", "").lower() if zoho_fields else ""
    if ("trader" in text or syndicator == "trader") and ("used" in text and "new" in text):
        return "E55"
    if re.search(r"(stock number|stock#).*[<>\;'\\\"]", text):
        return "E44"
    if "firewall" in text or "your request was rejected by d2c media's firewall" in text:
        return "E74"
    return ""

def classify_ticket_llm(ticket_message, context=None, model="gpt-4o"):
    FEWSHOT = """Example:
Message:
"Hi Véronique, Mazda Steele is still showing vehicles that were sold last week. Request to check the PBS import."
Zoho Fields:
contact: Véronique Fournier
dealer_name: Mazda Steele
dealer_id: 2618
rep: Véronique Fournier
category: Problem / Bug
sub_category: Import
syndicator: PBS
inventory_type:
"""
    system_prompt = (
        "You are a Zoho Desk classification assistant. Only use these allowed dropdown values for each field:\n"
        "Category: Product Activation – New Client, Product Activation – Existing Client, Product Cancellation, Problem / Bug, General Question, Analysis / Review, Other.\n"
        "Sub Category: Import, Export, Sales Data Import, FB Setup, Google Setup, Other Department, Other, AccuTrade.\n"
        "Inventory Type: New, Used, Demo, New + Used, or blank.\n"
        "If a value is not clear, leave it blank.\n" + FEWSHOT + "\nNow classify this message:"
    )
    user_prompt = f"""{ticket_message}

Return a JSON object:
{{
  "zoho_fields": {{
    "contact": ...,
    "dealer_name": ...,
    "dealer_id": ...,
    "rep": ...,
    "category": ...,
    "sub_category": ...,
    "syndicator": ...,
    "inventory_type": ...
  }},
  "zoho_comment": "...",
  "suggested_reply": "..."
}}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    match = re.search(r'{.*}', content, re.DOTALL)
    if not match:
        raise ValueError(f"❌ No JSON block found in response:\n{content}")

    # ✅ Fixed position — now always parsed
    result = json.loads(match.group(0))

    # ✅ Format Zoho Comment with structured content
    zf = result.get("zoho_fields", {})
    comment_parts = []
    if zf.get("dealer_name"):
        comment_parts.append(zf["dealer_name"])
    if zf.get("category"):
        comment_parts.append(zf["category"])
    if zf.get("sub_category"):
        comment_parts.append(f"Issue: {zf['sub_category']}")
    if zf.get("syndicator"):
        comment_parts.append(f"Syndicator: {zf['syndicator']}")
    comment_parts.append("Will investigate.")
    result["zoho_comment"] = "\n".join(comment_parts)

    return result

def classify_ticket(ticket_message):
    context = preprocess_ticket(ticket_message)
    fields = {}

    dealer_name, rep, dealer_id = extract_best_dealer(context, ticket_message)
    if dealer_id and rep and dealer_name:
        fields["dealer_name"] = dealer_name
        fields["dealer_id"] = dealer_id
        fields["rep"] = rep
        fields["contact"] = rep
    else:
        fields["dealer_name"] = ""
        fields["dealer_id"] = ""
        fields["rep"] = ""
        fields["contact"] = ""

    fields["syndicator"] = extract_syndicator(ticket_message)
    fields["inventory_type"] = next(
        (k.capitalize() for k in ["new + used", "new", "used", "demo"] if k in ticket_message.lower()), ""
    )
    fields["category"] = ""
    fields["sub_category"] = ""

    needs_llm = not fields["category"] or not fields["sub_category"]
    if needs_llm:
        result = classify_ticket_llm(ticket_message, context=fields)
        if dealer_id and rep and dealer_name:
            for key in ["dealer_name", "dealer_id", "rep", "contact"]:
                result["zoho_fields"][key] = fields[key]
        else:
            for key in ["dealer_name", "dealer_id", "rep", "contact"]:
                result["zoho_fields"][key] = ""
        for k in ["syndicator", "inventory_type"]:
            if not result["zoho_fields"].get(k):
                result["zoho_fields"][k] = fields.get(k, "")
        result["zoho_fields"] = validate_fields(result["zoho_fields"])
        result["edge_case"] = detect_edge_case(ticket_message, result["zoho_fields"])
        write_log(ticket_message, result, result["edge_case"])
        return result
    else:
        fields = validate_fields(fields)
        result = {
            "zoho_fields": fields,
            "zoho_comment": "",
            "suggested_reply": "",
            "edge_case": detect_edge_case(ticket_message, fields)
        }
        write_log(ticket_message, result, result["edge_case"])
        return result

if __name__ == "__main__":
    import sys
    msg = sys.stdin.read()
    result = classify_ticket(msg)
    print(json.dumps(result, indent=2, ensure_ascii=False))
