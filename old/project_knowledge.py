import re
import pandas as pd

SYNDICATOR_KEYWORDS = {}
try:
    df = pd.read_csv("Full_Syndicator_Keyword_Reference.csv")
    for value in df["Syndicator"].dropna():
        keyword = value.strip().lower()
        if keyword:
            if value not in SYNDICATOR_KEYWORDS:
                SYNDICATOR_KEYWORDS[value] = set()
            SYNDICATOR_KEYWORDS[value].add(keyword)
except Exception as e:
    print("Failed to load syndicator reference file:", e)

def classify_category_and_subcategory(text):
    lowered = text.lower()
    image_bug_keywords = [
        "image", "photo", "picture", "slide", "revert", "rearrange",
        "order changed", "removed", "lost", "randomly disappears",
        "certified image", "certified photo", "overwritten"
    ]
    if any(kw in lowered for kw in image_bug_keywords):
        return "Problem / Bug", "Import"

    if any(x in lowered for x in ["not receiving", "missing", "not showing", "disappeared", "not visible", "not appearing"]):
        if "export" in lowered or "pushed" in lowered:
            return "Problem / Bug", "Export"
        return "Problem / Bug", "Import"

    if any(x in lowered for x in ["cancel", "stop", "deactivate", "terminate"]):
        return "Product Cancellation", "Export"

    if re.search(r'\b(activate|setup|configure|create)\s+(an|the|our)?\s*export\b', lowered):
        return "Product Activation – Existing Client", "Export"

    if re.search(r'\bwhich\s+(imports|one is active)\b', lowered):
        return "General Question", "Import"

    return "General Question", "Other"

def extract_syndicator(text):
    return ""  # Handled by spaCy

def extract_inventory_type(text):
    lowered = text.lower()
    if any(x in lowered for x in ["certified", "used", "pre-owned", "usagées"]):
        return "Used"
    if any(x in lowered for x in ["new", "neufs"]):
        return "New"
    if "demo" in lowered:
        return "Demo"
    if "powersports" in lowered:
        return "Powersports"
    if "both" in lowered:
        return "Both"
    return ""

def extract_stock_numbers(text):
    return re.findall(r"stock[\s:#-]*([0-9A-Z]{4,})", text, re.IGNORECASE)