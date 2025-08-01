import os
import re
import json
import openai
import pandas as pd
from dealer_utils import preprocess_ticket
from datetime import datetime

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load dealer–rep mapping
mapping_df = pd.read_csv("rep_dealer_mapping.csv")
mapping_df["Dealer Name"] = mapping_df["Dealer Name"].str.lower().str.strip()
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
    synd = (zoho_fields or {}).get("syndicator","").lower()
    if ("trader" in text or synd=="trader") and "used" in text and "new" in text:
        return "E55"
    if re.search(r"(stock number|stock#).*?[<>'\"\\\\]", text):
        return "E44"
    if "firewall" in text:
        return "E74"
    if "partial" in text and "trim" in text and "inventory+" in text and "omni" in text:
        return "E77"
    return ""

def classify_ticket(text: str, model="gpt-4o"):
    # Preprocess context
    context = preprocess_ticket(text)

    # Build prompts
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
        "You are a Zoho Desk classification assistant. Only use these allowed dropdown values for each field:\n"
        "Category: Product Activation – New Client, Product Activation – Existing Client, Product Cancellation, Problem / Bug, General Question, Analysis / Review, Other.\n"
        "Sub Category: Import, Export, Sales Data Import, FB Setup, Google Setup, Other Department, Other, AccuTrade.\n"
        "Inventory Type: New, Used, Demo, New + Used, or blank.\n"
        + FEMSHOT +
        "\nNow classify this message and return ONLY the JSON object (no extra text)."
    )
    USER_PROMPT = f"""
Message:
{text}

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
}}
"""

    # Call OpenAI
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": USER_PROMPT},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()

    # Strip Markdown fences and extract JSON block
    # 1) Remove leading/trailing ```json or ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # 2) Extract JSON {...}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("❌ LLM did not return valid JSON:\n" + raw)
    json_text = m.group(0)

    # Parse JSON
    data = json.loads(json_text)

    # Post-process: fill rep and dealer_id via mapping if blank
    zf = data.get("zoho_fields", {})
    dn = zf.get("dealer_name", "").strip()
    if dn:
        lookup = lookup_dealer_by_name(dn)
        if not zf.get("rep"):
            zf["rep"] = lookup["rep"]
        if not zf.get("dealer_id"):
            zf["dealer_id"] = lookup["dealer_id"]

    # Edge case detection
    data["edge_case"] = detect_edge_case(text, zf)

    return data

# Optional batch runner
def batch_preprocess_csv(path="classifier_input_examples.csv"):
    df = pd.read_csv(path)
    for row in df.itertuples(index=False):
        print(f"--- Source: {row.source} ---")
        result = classify_ticket(row.message)
        print(json.dumps(result, indent=2))
        print()
    return
