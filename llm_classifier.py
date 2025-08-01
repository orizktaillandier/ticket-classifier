import os
import re
import json
import pandas as pd
from openai import OpenAI
from dealer_utils import preprocess_ticket, lookup_dealer_by_name, format_zoho_comment, detect_edge_case
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mapping_df = pd.read_csv("rep_dealer_mapping.csv")
mapping_df["Dealer Name"] = mapping_df["Dealer Name"].astype(str).str.lower().str.strip()
dealer_to_rep = mapping_df.set_index("Dealer Name")["Rep Name"].to_dict()
dealer_to_id  = mapping_df.set_index("Dealer Name")["Dealer ID"].to_dict()

def classify_ticket(text: str, model="gpt-4o"):
    context = preprocess_ticket(text)
    dealer_list = context.get("dealers_found", [])
    example = dealer_list[0] if dealer_list else find_example_dealer(text)
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
        "Important logic rules:\n"
        "- Only use real dealership rooftops as dealer_name (not group names like 'Kot Auto Group')\n"
        "- If a group name is used, try to extract the actual rooftop from examples or filenames\n"
        "- Never use 'Olivier Rizk-Taillandier' as rep unless the sender is actually him\n"
        "- The 'syndicator' field must refer to the export target (where D2C is sending the feed), not the data source or origin (e.g. Inventory+, PBS, SERTI)\n"
        "- If any field is uncertain or missing, leave it blank — logic will complete it\n"
        "- Do not infer — only return grounded field values\n"
        + FEMSHOT +
        "\nNow classify the following message and return ONLY the JSON object (no explanation, no extra text):"
    )

    USER_PROMPT = f"""
Message:
{text}

Return a JSON object exactly as follows:
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
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*\{", "{", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("❌ LLM did not return valid JSON:\n" + raw)
    json_text = m.group(0)
    data = json.loads(json_text)
    zf = data.get("zoho_fields", {})

    # --- Dealer name normalization & mapping ---
    # 1. Normalize the LLM output dealer_name (if present)
    dn_raw = zf.get("dealer_name", "")
    dn = re.sub(r"([a-z])([A-Z])", r"\1 \2", dn_raw).lower().strip()
    mapped_id = dealer_to_id.get(dn, "")
    mapped_rep = dealer_to_rep.get(dn, "")

    # 2. If not found, try example/fallback
    if not mapped_id and example:
        fallback_override = lookup_dealer_by_name(example)
        if fallback_override:
            zf["dealer_name"] = example.title()
            zf["dealer_id"] = fallback_override["dealer_id"]
            zf["rep"] = fallback_override["rep"]
    else:
        # 3. If mapped, override the LLM output with repmap values
        if mapped_id:
            zf["dealer_id"] = mapped_id
        if mapped_rep:
            zf["rep"] = mapped_rep

    # 4. If not found at all, leave blank (never guess)
    # 5. Use first detected rep as contact unless direct from client (not handled here)
    zf["dealer_name"] = zf.get("dealer_name", "").title()
    zf["contact"] = zf.get("rep", "")

    # 6. Syndicator: If not found but detected by logic, use that (never hardcode)
    if not zf.get("syndicator") and context.get("syndicators"):
        zf["syndicator"] = context["syndicators"][0].title()

    # 7. Compose the comment, edge case
    data["zoho_comment"] = format_zoho_comment(zf, context)
    data["edge_case"] = detect_edge_case(text, zf)
    data.pop("suggested_reply", None)

    return data

def find_example_dealer(text: str):
    patterns = [
        r"for ([A-Za-z0-9 &\\-]+)\\b",
        r"from ([A-Za-z0-9 &\\-]+)\\b",
        r"regarding ([A-Za-z0-9 &\\-]+)\\b"
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""
