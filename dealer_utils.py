import re
import nltk
import csv
import pandas as pd

nltk.download("punkt", quiet=True)

DEALER_BLOCKLIST = {"blue admin", "admin blue", "admin red", "d2c media", "cars commerce"}

def detect_language(text):
    return "fr" if re.search(r"\b(merci|bonjour|véhicule|images|depuis)\b", text.lower()) else "en"

def detect_stock_number(text):
    return bool(re.search(r"\b[A-Z0-9]{6,}\b", text))

def extract_contacts(text):
    lines = text.strip().split('\n')
    for i, line in enumerate(lines[-10:]):
        line = line.strip()
        if re.match(r'^(Best regards|Regards|Merci|Thanks|Cordially|From:|Envoyé par|De:)', line, re.IGNORECASE):
            next_idx = i + 1
            if next_idx < len(lines[-10:]):
                next_line = lines[-10:][next_idx].strip()
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
    dealer_matches = re.findall(
        r"\b(?:mazda|toyota|honda|chevrolet|hyundai|genesis|ford|ram|gmc|acura|jeep"
        r"|buick|nissan|volvo|subaru|volkswagen|kia|mitsubishi|infiniti|lexus"
        r"|cadillac|dodge|mini|jaguar|land rover|bmw|mercedes|audi|porsche|tesla)"
        r"[a-zé\-\s]*", text.lower()
    )
    INVALID_SUFFIXES = {"units", "inventory", "vehicles", "images", "stock"}

    cleaned = []
    for d in dealer_matches:
        d_clean = d.strip()
        if d_clean in DEALER_BLOCKLIST:
            continue
        parts = d_clean.split()
        if parts and parts[-1] not in INVALID_SUFFIXES:
            d_clean = re.sub(r"([a-z])([A-Z])", r"\1 \2", d_clean)  # split camel case
            cleaned.append(d_clean.strip())

    return list(set(cleaned))

def extract_syndicators(text):
    candidates = [
        "vauto", "easydeal", "car media", "icc", "homenet", "serti",
        "evolutionautomobiles", "spincar", "trader", "pbs", "omni"
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

def batch_preprocess_csv(path="classifier_input_examples.csv"):
    df = pd.read_csv(path)
    results = []
    for row in df.itertuples(index=False):
        context = preprocess_ticket(row.message)
        context["source"] = row.source
        results.append(context)
    return results

def lookup_dealer_by_name(name, csv_path="rep_dealer_mapping.csv"):
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower().strip()
    df = pd.read_csv(csv_path)
    df["Dealer Name"] = df["Dealer Name"].str.lower().str.strip()

    match = df[df["Dealer Name"] == name]
    if not match.empty:
        return {
            "dealer_id": match.iloc[0]["Dealer ID"],
            "rep": match.iloc[0]["Rep Name"]
        }
    return {}
