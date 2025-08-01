import os
import pandas as pd
import nltk
from dotenv import load_dotenv
from llm_classifier import classify_ticket, write_log

# Setup
load_dotenv()
nltk.download("punkt", quiet=True)

def classify_batch_from_csv(path="Classifier_Complex_Input_Examples.csv"):
    df = pd.read_csv(path)
    msg_col = "message" if "message" in df.columns else df.columns[0]
    print(f"Detected message column: {msg_col}\n")

    for i, row in df.iterrows():
        ticket_message = str(row[msg_col])
        print(f"\n📩 Ticket {i+1} Source: Email from client\n")
        result = classify_ticket(ticket_message)
        fields = result.get("zoho_fields", {})

        print("📋 Summary for Zoho Fields:")
        for field, label in [
            ("contact", "Contact"),
            ("dealer_name", "Dealer Name"),
            ("dealer_id", "Dealer ID"),
            ("rep", "Rep"),
            ("category", "Category"),
            ("sub_category", "Sub Category"),
            ("syndicator", "Syndicator"),
            ("inventory_type", "Inventory Type"),
        ]:
            print(f"{label:<15}: {fields.get(field, '')}")

        print("\n📝 Zoho Comment:")
        print(result.get("zoho_comment", "").strip())

        print("\n✉️ Suggested Reply:")
        print(result.get("suggested_reply", "").strip())
        edge_case = result.get("edge_case", "")
        if edge_case:
            print(f"\n⚠️  Edge Case Flagged: {edge_case}")

        print("=" * 80)
        write_log(ticket_message, result, edge_case)

if __name__ == "__main__":
    classify_batch_from_csv()
