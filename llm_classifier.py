import re
import openai
import json
from dealer_utils import lookup_dealer_by_name

client = openai.OpenAI()

def validate_fields(fields):
    valid = {
        "category": [
            "Product Activation – New Client",
            "Product Activation – Existing Client",
            "Product Cancellation",
            "Problem / Bug",
            "General Question",
            "Analysis / Review",
            "Other",
        ],
        "sub_category": [
            "Import",
            "Export",
            "Sales Data Import",
            "FB Setup",
            "Google Setup",
            "Other Department",
            "Other",
            "AccuTrade",
        ],
        "inventory_type": [
            "New",
            "Used",
            "Demo",
            "New + Used",
            "",
        ],
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
    if re.search(r"stock number[\\s:]*[<>;\'\"-]", text):
        return "E44"
    if "firewall" in text or "your request was rejected by d2c media's firewall" in text:
        return "E74"
    if "partial" in text and "trim" in text and "inventory+" in text and "omni" in text:
        return "E77"
    return ""

def classify_ticket_llm(ticket_message, context=None, model="gpt-4o"):
    FEMSHOT = """
Example:
Message:
Hi Véronique, Mazda Steele is still showing vehicles that were sold last week. Request to check the PBS import.

Zoho Fields:
"contact": "Véronique Fournier"
"dealer_name": "Mazda Steele"
"dealer_id": "2618"
"rep": "Véronique Fournier"
"category": "Problem / Bug"
"sub_category": "Import"
"syndicator": "PBS"
"inventory_type": ""

# Notes:
# If a group name is mentioned (e.g., "Kot Auto Group") and another dealer (e.g., "Maple Ridge Hyundai") is cited as an example or export file,
# always assign the Zoho fields based on the main group or requestor, not the example.
"""

    system_prompt = (
        "You are a Zoho Desk classification assistant. Only use these allowed dropdown values for each field:\n"
        "Category: Product Activation – New Client, Product Activation – Existing Client, Product Cancellation, "
        "Problem / Bug, General Question, Analysis / Review, Other.\n"
        "Sub Category: Import, Export, Sales Data Import, FB Setup, Google Setup, Other Department, Other, AccuTrade.\n"
        "Inventory Type: New, Used, Demo, New + Used, or blank.\n"
        + FEMSHOT +
        "\nNow classify this message:"
    )

    user_prompt = ticket_message + "\n\nReturn a JSON object:\n\n" \
        "{\n" \
        "  \"zoho_fields\": {\n" \
        "    \"contact\": \"...\",\n" \
        "    \"dealer_name\": \"...\",\n" \
        "    \"dealer_id\": \"...\",\n" \
        "    \"rep\": \"...\",\n" \
        "    \"category\": \"...\",\n" \
        "    \"sub_category\": \"...\",\n" \
        "    \"syndicator\": \"...\",\n" \
        "    \"inventory_type\": \"...\"\n" \
        "  },\n" \
        "  \"zoho_comment\": \"...\",\n" \
        "  \"suggested_reply\": \"...\"\n" \
        "}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    parsed = json.loads(response.choices[0].message.content)
    parsed["zoho_fields"] = validate_fields(parsed.get("zoho_fields", {}))

    # Override rep + ID if mapping match is found
    name = parsed["zoho_fields"].get("dealer_name", "").strip()
    if name:
        match = lookup_dealer_by_name(name)
        if match:
            parsed["zoho_fields"]["dealer_id"] = match["dealer_id"]
            parsed["zoho_fields"]["rep"] = match["rep"]

    # Detect edge case
    parsed["edge_case"] = detect_edge_case(ticket_message, parsed.get("zoho_fields"))

    return parsed
