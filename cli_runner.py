import sys
from llm_classifier import classify_ticket

if __name__ == "__main__":
    print("\U0001f4e8 Paste your ticket message below. Press Ctrl+D (Linux/macOS) or Ctrl+Z (Windows) when done:\n")
    message = sys.stdin.read().strip()

    print("\n\U0001f4c4 Output:")
    print("=" * 60)
    result = classify_ticket(message)
    for field in [
        "contact", "dealer_name", "dealer_id", "rep",
        "category", "sub_category", "syndicator", "inventory_type"
    ]:
        val = result.get("zoho_fields", {}).get(field, "")
        print(f"{field.title():<15}: {val}")
    print("\n📝 Zoho Comment:")
    print(result.get("zoho_comment", "").strip())
    print("\n✉️ Suggested Reply:")
    print(result.get("suggested_reply", "").strip())
    edge_case = result.get("edge_case", "")
    if edge_case:
        print(f"\n⚠️  Edge Case Flagged: {edge_case}")
    print("=" * 60)

    input("\n✅ Press Enter to exit.")
