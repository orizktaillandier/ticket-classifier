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

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def find_example_dealer(ticket_message):
    """
    Looks for patterns like 'for Maple Ridge Hyundai' as the example dealer for the ticket.
    Returns the best-matched dealer from mapping if found, else (None, None, None).
    """
    # Looks for phrases like "...for Maple Ridge Hyundai", etc.
    matches = re.findall(r'for ([A-Za-z0-9 &\-\']+)', ticket_message)
    if matches:
        for candidate in matches:
            dealer_name, rep, dealer_id = robust_dealer_match([candidate])
            if dealer_id:
                return dealer_name, rep, dealer_id
    return None, None, None

def extract_syndicator(text):
    known = [
        "vauto", "easydeal", "car media", "icc", "homenet", "serti",
        "evolutionautomobiles", "spincar", "trader", "pbs", "omni"
    ]
    lower = text.lower()
    # Prefer "to [syndicator]" pattern for export target
    match = re.search(r'to ([a-z0-9 .\-]+)', lower)
    if match:
        candidate = match.group(1).strip()
        for s in known:
            if s in candidate:
                return s.title() if s != "icc" else "ICC"
    # Fallback to previous logic
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
    if re.search(r"(stock number|stock#).*?[<>\'\"\\]", text):
        return "E44"
    if "firewall" in text or "your request was rejected by d2c media's firewall" in text:
        return "E74"
    if "partial" in text and "trim" in text and "inventory+" in text and "omni" in text:
        return "E77"
    return ""

def classify_ticket_llm(ticket_message, context=None, model="gpt-4o"):
    FEWSHOT = (
        "Example:\n"
        "Message:\n"
        "\"Hi Véronique, Mazda Steele is still showing vehicles that were sold last week. "
        "Request to check the PBS import.\"\n"
        "Zoho Fields:\n"
        "contact: Véronique Fournier\n"
        "dealer_name: Mazda Steele\n"
        "dealer_id: 2618\n"
        "rep: Véronique Fournier\n"
        "category: Problem / Bug\n"
        "sub_category: Import\n"
        "syndicator: PBS\n"
        "inventory_type:\n"
    )

    system_prompt = (
        "You are a Zoho Desk classification assistant. Only use these allowed dropdown values for each field:\n"
        "Category: Product Activation – New Client, Product Activation – Existing Client, Product Cancellation, "
        "Problem / Bug, General Question, Analysis / Review, Other.\n"
        "Sub Category: Import, Export, Sales Data Import, FB Setup, Google Setup, Other Department, Other, AccuTrade.\n"
        "Inventory Type: New, Used, Demo, New + Used, or blank.\n" + FEWSHOT +
        "\nNow classify this message:"
    )

    user_prompt = f"""{ticket_message}

Return a JSON object:
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
}}"""

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
    result = json.loads(match.group(0))

    # Use LLM-generated Zoho comment as-is
    result["zoho_comment"] = result.get("zoho_comment", "").strip()

    return result

def classify_ticket(ticket_message):
    context = preprocess_ticket(ticket_message)
    fields = {}

    # Enhanced: Try to extract "example" dealer from text context first
    dealer_name, rep, dealer_id = find_example_dealer(ticket_message)
    if not dealer_id:
        dealer_name, rep, dealer_id = extract_best_dealer(context, ticket_message)

    if dealer_id and rep and dealer_name:
        fields["dealer_name"] = dealer_name
        fields["dealer_id"] = dealer_id
        fields["rep"] = rep
        # Only assign client name if it's a valid dealer email
        if re.search(r"@(kotautogroup\.com|dealer\.com|auto\.ca|cars\.com)$", ticket_message.lower()):
            fields["contact"] = context.get("sender_name", rep)
        else:
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
        for key in ["dealer_name", "dealer_id", "rep", "contact"]:
            if fields.get(key):
                result["zoho_fields"][key] = fields[key]
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
