import os
import openai
import json
from dealer_utils import preprocess_ticket
import pandas as pd

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load mapping
mapping_df = pd.read_csv("rep_dealer_mapping.csv")
mapping_df["Dealer Name"] = mapping_df["Dealer Name"].str.lower()
dealer_to_rep = mapping_df.set_index("Dealer Name")["Rep Name"].to_dict()
dealer_to_id = mapping_df.set_index("Dealer Name")["Dealer ID"].to_dict()

def lookup_dealer_by_name(name):
    name_lower = name.lower()
    return {
        "rep": dealer_to_rep.get(name_lower, ""),
        "dealer_id": dealer_to_id.get(name_lower, "")
    }

def classify_ticket(text, model="gpt-4o"):
    context = preprocess_ticket(text)

    # FEMSHOT injection for debugging
    femshot = (
        "EXAMPLE:\n"
        "Message:\n"
        "\"Hi Véronique, Mazda Steele is still showing vehicles that were sold last week. Request to check the PBS import.\"\n\n"
        "Zoho Fields:\n"
        "{\n"
        "  \"contact\": \"Véronique Fournier\",\n"
        "  \"dealer_name\": \"Mazda Steele\",\n"
        "  \"dealer_id\": \"2618\",\n"
        "  \"rep\": \"Véronique Fournier\",\n"
        "  \"category\": \"Problem / Bug\",\n"
        "  \"sub_category\": \"Import\",\n"
        "  \"syndicator\": \"PBS\",\n"
        "  \"inventory_type\": \"\"\n"
        "}\n"
        "\"zoho_comment\": \"mazda steele (2618)\nrep: véronique fournier\npbs import – issue with stale vehicles. will check.\",\n"
        "\"suggested_reply\": \"Hi Véronique,\n\nThanks for reaching out.\nI will take a look and follow up shortly.\n\nLet me know if there is anything else.\n\nThanks,\nOlivier\""
    )

    SYSTEM_PROMPT = (
        "You are a Zoho Desk classification assistant. Only use these allowed dropdown values for each field:\n"
        "Category: Product Activation – New Client, Product Activation – Existing Client, Product Cancellation, Problem / Bug, General Question, Analysis / Review, Other.\n"
        "Sub Category: Import, Export, Sales Data Import, FB Setup, Google Setup, Other Department, Other, AccuTrade.\n"
        "Inventory Type: New, Used, Demo, New + Used, or blank.\n"
        f"\n{femshot}\n\nNow classify this message."
    )

    USER_PROMPT = (
        "Return a JSON object:\n\n"
        "{\n"
        "  \"zoho_fields\": {\n"
        "    \"contact\": \"...\",\n"
        "    \"dealer_name\": \"...\",\n"
        "    \"dealer_id\": \"...\",\n"
        "    \"rep\": \"...\",\n"
        "    \"category\": \"...\",\n"
        "    \"sub_category\": \"...\",\n"
        "    \"syndicator\": \"...\",\n"
        "    \"inventory_type\": \"...\"\n"
        "  },\n"
        "  \"zoho_comment\": \"...\",\n"
        "  \"suggested_reply\": \"...\"\n"
        "}\n\n"
        f"Message:\n{text}"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("❌ LLM did not return valid JSON:\n\n" + raw)

    # Postprocess rep and ID if missing
    dealer_name = parsed.get("zoho_fields", {}).get("dealer_name", "")
    if dealer_name:
        lookup = lookup_dealer_by_name(dealer_name)
        if parsed["zoho_fields"].get("rep", "") == "":
            parsed["zoho_fields"]["rep"] = lookup["rep"]
        if parsed["zoho_fields"].get("dealer_id", "") == "":
            parsed["zoho_fields"]["dealer_id"] = lookup["dealer_id"]

    return parsed

def batch_preprocess_csv(path="classifier_input_examples.csv"):
    df = pd.read_csv(path)
    results = []
    for row in df.itertuples(index=False):
        print(f"\n--- Ticket Source: {row.source} ---")
        parsed = classify_ticket(row.message)
        print(json.dumps(parsed, indent=2))
        results.append(parsed)
    return results
