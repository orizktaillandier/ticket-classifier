import openai
import os
import json
import pandas as pd
from preprocessor import preprocess_ticket, batch_preprocess_csv
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a ticket classification assistant. Based on the message and context, return:

- Zoho Fields: Contact, Dealer Name, Dealer ID, Rep, Category, Sub Category, Syndicator, Inventory Type
- Zoho Comment: short, field-based
- Suggested Reply: matches D2C Media tone and template rules

Only use known dropdowns for Category, Sub Category, and Inventory Type. If a field is not inferable, leave it blank.
Avoid assumptions. Match how a real support analyst would reason.
"""

# Load dealer-rep mapping
mapping_df = pd.read_csv("rep_dealer_mapping.csv")
mapping_df["Dealer Name"] = mapping_df["Dealer Name"].str.lower()
dealer_to_rep = mapping_df.set_index("Dealer Name")["Rep Name"].to_dict()
dealer_to_id = mapping_df.set_index("Dealer Name")["Dealer ID"].to_dict()

def build_prompt(text, context):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ticket message:\n{text}\n\nExtracted context:\n{json.dumps(context, indent=2)}"}
    ]

def classify_ticket_llm(text):
    context = preprocess_ticket(text)
    messages = build_prompt(text, context)
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.2,
        messages=messages
    )
    result = response.choices[0].message.content

    # Auto-fill Dealer ID and Rep if Dealer Name matches confidently
    for line in result.splitlines():
        if line.lower().startswith("dealer name:"):
            dealer_name = line.split(":", 1)[1].strip().lower()
            if dealer_name in dealer_to_rep:
                result = result.replace("Rep:", f"Rep: {dealer_to_rep[dealer_name]}")
            if dealer_name in dealer_to_id:
                result = result.replace("Dealer ID:", f"Dealer ID: {dealer_to_id[dealer_name]}")
            break

    return result

def classify_batch_from_csv(path="classifier_input_examples.csv"):
    df = pd.read_csv(path)
    for row in df.itertuples(index=False):
        print(f"--- Ticket Source: {row.source} ---")
        result = classify_ticket_llm(row.message)
        print(result)
        print("=" * 80)

if __name__ == "__main__":
    classify_batch_from_csv()
