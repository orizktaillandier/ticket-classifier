import os
import re
import json
import pandas as pd
from openai import OpenAI
from dealer_utils import preprocess_ticket, lookup_dealer_by_name, format_zoho_comment, detect_edge_case
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load mapping CSV once at module load
mapping_df = pd.read_csv("rep_dealer_mapping.csv")
mapping_df["Dealer Name"] = mapping_df["Dealer Name"].astype(str).str.lower().str.strip()
dealer_to_rep = mapping_df.set_index("Dealer Name")["Rep Name"].to_dict()
dealer_to_id  = mapping_df.set_index("Dealer Name")["Dealer ID"].to_dict()

def classify_ticket(text: str, model="gpt-4o"):
    context = preprocess_ticket(text)
    dealer_list = context.get("dealers_found", [])
    # Always use all possible dealer names: LLM output, context, and fallback
    dealer_candidates = []

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
        "- Return a JSON object with ALL keys present and fill missing keys with empty string.\n"
        "- Do NOT include any explanations or extra text outside the JSON.\n"
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

    # --- LLM call + error handling ---
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
        # Uncomment for debugging
        # print("RAW LLM OUTPUT:", raw)
    except Exception as e:
        print("❌ LLM call failed:", repr(e))
        return {"error": str(e)}

    # --- Parse JSON output from LLM ---
    raw = re.sub(r"^```(?:json)?\s*\{", "{", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("❌ LLM did not return valid JSON:\n" + raw)
    json_text = m.group(0)
    data = json.loads(json_text)
    zf = data.get("zoho_fields", {})

    # --- Always build list of all possible dealer name candidates ---
    # LLM output first, then context detections, then fallback example
    dn_llm = zf.get("dealer_name", "").strip()
    if dn_llm:
        dealer_candidates.append(dn_llm)
    for d in dealer_list:
        d = d.strip()
        if d and d not in dealer_candidates:
            dealer_candidates.append(d)
    # Last-resort: use fallback from "for X", "from X", "regarding X"
    fallback = find_example_dealer(text)
    if fallback and fallback not in dealer_candidates:
        dealer_candidates.append(fallback)

    # --- Try mapping each candidate until success ---
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

    # --- Fallback to lookup_dealer_by_name if no direct match ---
    if not matched_id and dealer_candidates:
        for name in dealer_candidates:
            override = lookup_dealer_by_name(name)
            if override.get("dealer_id"):
                matched_name = name
                matched_id = override["dealer_id"]
                matched_rep = override["rep"]
                break

    # --- Update output with mapping, only if we find a valid match ---
    if matched_id:
        zf["dealer_name"] = matched_name.title()
        zf["dealer_id"] = matched_id
        zf["rep"] = matched_rep
        zf["contact"] = matched_rep
    else:
        # If LLM output for rep is valid, use it as contact, otherwise blank
        zf["dealer_name"] = dn_llm.title() if dn_llm else ""
        zf["contact"] = zf.get("rep", "")

    # Ensure all keys exist as strings
    expected_keys = [
        "contact", "dealer_name", "dealer_id", "rep",
        "category", "sub_category", "syndicator", "inventory_type"
    ]
    for key in expected_keys:
        if key not in zf or not isinstance(zf[key], str):
            zf[key] = ""

    # --- Syndicator fallback if missing and context finds one ---
    if not zf.get("syndicator") and context.get("syndicators"):
        zf["syndicator"] = context["syndicators"][0].title()

    # --- Compose comment and edge case ---
    data["zoho_comment"] = format_zoho_comment(zf, context)
    data["edge_case"] = detect_edge_case(text, zf)
    data.pop("suggested_reply", None)

    return data

def find_example_dealer(text: str):
    patterns = [
        r"for ([A-Za-z0-9 &\-\']+)\b",
        r"from ([A-Za-z0-9 &\-\']+)\b",
        r"regarding ([A-Za-z0-9 &\-\']+)\b"
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""
