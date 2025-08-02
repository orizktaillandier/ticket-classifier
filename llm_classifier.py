import os
import re
import json
import pandas as pd
from openai import OpenAI
from dealer_utils import preprocess_ticket, lookup_dealer_by_name, format_zoho_comment, detect_edge_case
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    mapping_df = pd.read_csv("rep_dealer_mapping.csv")
    mapping_df["Dealer Name"] = mapping_df["Dealer Name"].astype(str).str.lower().str.strip()
    dealer_to_rep = mapping_df.set_index("Dealer Name")["Rep Name"].to_dict()
    dealer_to_id  = mapping_df.set_index("Dealer Name")["Dealer ID"].to_dict()
except Exception as e:
    # Log and crash clearly
    raise RuntimeError(f"❌ FATAL: Could not load 'rep_dealer_mapping.csv'. Reason: {e}")

def classify_ticket(text: str, model="gpt-4o"):
    context = preprocess_ticket(text)
    dealer_list = context.get("dealers_found", [])
    dealer_candidates = []

    FEMSHOT = """
Example 1:
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

Example 2:
Message:
"Hi, we’d like to activate a Car Media export for our new and used vehicles at Lallier Kia Laval."

Zoho Fields:
{
  "contact": "",
  "dealer_name": "Lallier Kia Laval",
  "dealer_id": "",
  "rep": "",
  "category": "Product Activation – Existing Client",
  "sub_category": "Export",
  "syndicator": "Car Media",
  "inventory_type": "New + Used"
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
        "- Return a JSON object with ALL keys present and fill missing keys with empty string\n"
        "- Do not include markdown (e.g. ```json) or explanation. Return only raw JSON\n"
        "- Do not invent or guess Dealer ID — use mapping or leave blank\n"
        + FEMSHOT +
        "\nNow classify the following message and return ONLY the JSON object as exactly specified:"
    )

    USER_PROMPT = f"""
Message:
{text}

Return a JSON object exactly as follows, with ALL keys present (use empty strings if unknown):
{{
  "zoho_fields": {{
    "contact": "",
    "dealer_name": "",
    "dealer_id": "",
    "rep": "",
    "category": "",
    "sub_category": "",
    "syndicator": "",
    "inventory_type": ""
  }},
  "zoho_comment": "",
  "suggested_reply": ""
}}
"""

    # Optional inline hints for edge cases
    if context.get("image_flags"):
        USER_PROMPT = "[Contains image/photo keywords]\n" + USER_PROMPT
    if context.get("contains_stock_number"):
        USER_PROMPT = "[Contains stock number]\n" + USER_PROMPT
    if dealer_list:
        SYSTEM_PROMPT = f"Detected dealer candidates: {', '.join(dealer_list)}\n\n" + SYSTEM_PROMPT

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        print("❌ LLM call failed:", repr(e))
        return {"error": str(e)}

    raw = re.sub(r"^```(?:json)?\s*\{", "{", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("❌ LLM did not return valid JSON:\n" + raw)
    json_text = m.group(0)
    data = json.loads(json_text)
    zf = data.get("zoho_fields", {})

    # All dealer matching logic unchanged
    dn_llm = zf.get("dealer_name", "").strip()
    if dn_llm:
        dealer_candidates.append(dn_llm)
    for d in dealer_list:
        d = d.strip()
        if d and d not in dealer_candidates:
            dealer_candidates.append(d)
    fallback = find_example_dealer(text)
    if fallback and fallback not in dealer_candidates:
        dealer_candidates.append(fallback)

    matched_name = ""
    matched_id = ""
    matched_rep = ""
    for name in dealer_candidates:
        norm = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower().strip()
        if norm in dealer_to_id:
            matched_name = name
            matched_id = dealer_to_id[norm]
            matched_rep = dealer_to_rep.get(norm, "")
            break

    if not matched_id and dealer_candidates:
        for name in dealer_candidates:
            override = lookup_dealer_by_name(name)
            if override.get("dealer_id"):
                matched_name = name
                matched_id = override["dealer_id"]
                matched_rep = override["rep"]
                break

    if matched_id:
        zf["dealer_name"] = matched_name.title()
        zf["dealer_id"] = matched_id
        zf["rep"] = matched_rep
        zf["contact"] = matched_rep
    else:
        # === Group fallback logic: if no rooftop match, check for group in mapping ===
        group_found = False
        for name, id_ in dealer_to_id.items():
            if "group" in name.lower():
                zf["dealer_name"] = name.title() + " (Group suggestion)"
                zf["dealer_id"] = id_
                zf["rep"] = ""
                zf["contact"] = ""
                group_found = True
                break
        if not group_found:
            zf["dealer_name"] = dn_llm.title() if dn_llm else ""
            zf["contact"] = zf.get("rep", "")

    expected_keys = [
        "contact", "dealer_name", "dealer_id", "rep",
        "category", "sub_category", "syndicator", "inventory_type"
    ]
    for key in expected_keys:
        if key not in zf or not isinstance(zf[key], str):
            zf[key] = ""

    if not zf.get("syndicator") and context.get("syndicators"):
        zf["syndicator"] = context["syndicators"][0].title()

    data["zoho_comment"] = format_zoho_comment(zf, context)
    data["edge_case"] = detect_edge_case(text, zf)
    data.pop("suggested_reply", None)

    return data

def find_example_dealer(text: str):
    patterns = [
        r"for ([A-Za-z0-9 &\\-\\']+)\\b",
        r"from ([A-Za-z0-9 &\\-\\']+)\\b",
        r"regarding ([A-Za-z0-9 &\\-\\']+)\\b"
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""
