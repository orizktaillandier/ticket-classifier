import re
import nltk
import pandas as pd

nltk.download("punkt", quiet=True)

DEALER_BLOCKLIST = {"blue admin", "admin blue", "admin red", "d2c media", "cars commerce"}

def detect_language(text):
    return "fr" if re.search(r"\b(merci|bonjour|véhicule|images|depuis)\b", text.lower()) else "en"

def detect_stock_number(text):
    return bool(re.search(r"\b[A-Z0-9]{6,}\b", text))

def extract_contacts(text):
    lines = text.strip().split('\n')
    for i in range(len(lines) - 1):
        line = lines[i].strip().lower()
        if re.match(r'^(best regards|regards|merci|thanks|cordially|from:|envoyé par|de:)', line, re.IGNORECASE):
            next_line = lines[i + 1].strip()
            name_match = re.match(r'^[A-Z][a-z]+( [A-Z][a-z]+)+$', next_line)
            if name_match:
                return next_line
    greet_match = re.search(r'^(hi|bonjour|hello|salut)[\s,:-]+([A-Z][a-z]+)', text.strip(), re.IGNORECASE | re.MULTILINE)
    if greet_match:
        candidate = greet_match.group(2)
        if not re.match(r'^(nous|client|dealer|photos?|images?|request|inventory)$', candidate, re.IGNORECASE):
            return candidate
    match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
    if match and not re.match(r'^(nous|client|dealer|photos?|images?|request|inventory)$', match.group(1), re.IGNORECASE):
        return match.group(1)
    return ""

def extract_dealers(text):
    # 1. Look for "Dealership Name: ..." or similar
    lines = text.split('\n')
    extracted = []
    for line in lines:
        m = re.search(r"(Dealership Name|Dealer Name|Dealer)\s*[:\-]?\s*([A-Za-z0-9 &'\-]+)", line, re.IGNORECASE)
        if m:
            candidate = m.group(2).strip()
            if candidate:
                extracted.append(candidate.lower())
    # 2. Fall back to old OEM matcher
    if not extracted:
        dealer_matches = re.findall(
            r"\b(?:mazda|toyota|honda|chevrolet|hyundai|genesis|ford|ram|gmc|acura|jeep"
            r"|buick|nissan|volvo|subaru|volkswagen|kia|mitsubishi|infiniti|lexus"
            r"|cadillac|dodge|mini|jaguar|land rover|bmw|mercedes|audi|porsche|tesla)"
            r"[a-zé\-\s]*\b", text.lower()
        )
        INVALID_SUFFIXES = {"units", "inventory", "vehicles", "images", "stock"}
        cleaned = []
        for d in dealer_matches:
            d_clean = d.strip()
            parts = d_clean.split()
            if parts and parts[-1] not in INVALID_SUFFIXES:
                cleaned.append(d_clean)
        extracted = list(set(cleaned))
    return extracted

def extract_syndicators(text):
    candidates = [
        "vauto", "easydeal", "car media", "icc", "homenet", "serti",
        "evolutionautomobiles", "spincar", "trader", "pbs", "google", "omni", "gubagoo"
    ]
    return [c for c in candidates if c in text.lower()]

def extract_image_flags(text):
    flags = []
    lower = text.lower()
    if "image" in lower:
        flags.append("image")
    if "certified" in lower:
        flags.append("certified")
    if "overwrite" in lower or "overwritten" in lower:
        flags.append("overwritten")
    return flags

def preprocess_ticket(text):
    return {
        "message": text,
        "contains_french": detect_language(text) == "fr",
        "contains_stock_number": detect_stock_number(text),
        "contacts_found": [extract_contacts(text)],
        "dealers_found": extract_dealers(text),
        "syndicators": extract_syndicators(text),
        "image_flags": extract_image_flags(text),
        "line_count": text.count("\n") + 1
    }

def lookup_dealer_by_name(name, csv_path="rep_dealer_mapping.csv"):
    name = name.lower().strip()
    df = pd.read_csv(csv_path)
    df["Dealer Name"] = df["Dealer Name"].str.lower().str.strip()

    match = df[df["Dealer Name"] == name]
    if not match.empty:
        return {
            "dealer_id": str(match.iloc[0]["Dealer ID"]),
            "rep": match.iloc[0]["Rep Name"]
        }
    return {}

def detect_edge_case(message: str, zoho_fields=None):
    text = message.lower()
    synd = (zoho_fields or {}).get("syndicator", "").lower()
    if ("trader" in text or synd == "trader") and "used" in text and "new" in text:
        return "E55"
    if re.search(r"(stock number|stock#).*?[<>'\\\\\"]", text):
        return "E44"
    if "firewall" in text:
        return "E74"
    if "partial" in text and "trim" in text and "inventory+" in text and "omni" in text:
        return "E77"
    return ""

def format_zoho_comment(zf, context):
    print("=== USING THE NEW UNIVERSAL COMMENT FUNCTION ===")
    lines = []
    category = zf.get('category', '').lower()
    sub_category = zf.get('sub_category', '').lower()
    syndicator = zf.get('syndicator', '')
    inventory_type = zf.get('inventory_type', '') or 'New + Used'

    # Dealer/rep
    lines.append(f"{zf.get('dealer_name', '')} ({zf.get('dealer_id', '')})")
    lines.append(f"Rep: {zf.get('rep', '')}")

    # Dealer contact: only show first non-D2C/CarsCommerce email found in message
    dealer_emails = []
    for e in re.findall(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", context.get("message", "").lower()):
        if not any(skip in e for skip in ("d2cmedia", "carscommerce")):
            dealer_emails.append(e)
    if dealer_emails:
        lines.append(f"Dealer contact: {dealer_emails[0]}")

    # ---- Export Activation ----
    if sub_category == "export":
        lines.append(f"Export: {syndicator} – {inventory_type}")
        lines.append("")
        lines.append("@Audrey Girard approuves-tu ce nouvel export?")
        lines.append("Merci!")

    # ---- Import/Sync Issue ----
    elif sub_category == "import":
        lines.append(f"Import: {syndicator} – {inventory_type}")
        lines.append("")
        lines.append("Client reports import/sync issue. Will investigate.")

    # ---- Image/Photo Bug ----
    elif "image" in context.get("image_flags", []) or "photo" in context.get("message", "").lower():
        lines.append("Client says issue with vehicle images/photos.")
        lines.append("Looks random.")
        lines.append("Will investigate.")

    # ---- System/Firewall Error ----
    elif "firewall" in context.get("message", "").lower():
        lines.append("Partner unable to pull import due to firewall block.")
        lines.append("Will escalate.")

    # ---- Default / Other ----
    else:
        lines.append("Ticket logged for review.")
        lines.append("Will investigate.")

    return "\n".join(lines)
