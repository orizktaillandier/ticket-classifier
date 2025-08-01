function_schema = [{
    "name": "classify_ticket",
    "description": "Classifies Zoho ticket into required fields",
    "parameters": {
        "type": "object",
        "properties": {
            "dealer_name": {"type": "string"},
            "dealer_id": {"type": "string"},
            "category": {"type": "string"},
            "sub_category": {"type": "string"},
            "syndicator": {"type": "string"},
            "inventory_type": {"type": "string"},
        },
        "required": ["dealer_name", "dealer_id", "category", "sub_category"]
    }
}]
