import re
import nltk
import csv
import pandas as pd

nltk.download("punkt", quiet=True)

DEALER_BLOCKLIST = {"blue admin", "admin blue", "admin red", "d2c media", "cars commerce"}

def detect_language(text):
    return "fr" if re.search(r"\\b(merci|bonjour|véhicule|images|depuis)\\b", text.lower()) else "en"

def detect_stock_number(text):
    return bool(re.search(r"\\b[A-Z0-9]{6,}\\b", text))

def extract_contacts(text):
    lines = text.strip().split('\\n')
    for i, line in enumerate(lines[-10:]):
        line = line.strip()
        if re.match(r'^(Best regards|Regards|Merci|Thanks|Cordially|From:|Envoyé par|De:)', line, re.IGNORECASE):
            try:
                next_line = lines[-10:][i + 1].strip()
                name_match = re.match(r'^[A-Z][a-z]+( [A-Z][a-z]+)+$', next_line)
                if name_match:
                    return next_line
            except IndexError:
                continue
    greet_match = re.search(r'^(hi|bonjour|hello|salut)[\\s,:-]+([A-Z][a-z]+)', text.strip(), re.IGNORECASE | re.MULTILINE)
    if greet_match:
        candidate = greet_match.group(2)
        if not re.match(r'^(nous|client|dealer|photos?|images?|request|inventory)$', candidate, re.IGNORECASE):
            return candidate
    match = re.search(r'\\b([A-Z][a-z]+ [A-Z][a-z]+)\\b', text)
    if match and not re.match(r'^(nous|client|dealer|photos?|images?|request|inventory)$', match.group(1), re.IGNORECASE):
        return match.group(1)
    return ""

def extract_dealers(text):
    dealer_matches = re.findall(
        r"\\b(?:mazda|toyota|honda|chevrolet|hyundai|genesis|ford|ram|gmc|acura|jeep"
        r"|buick|nissan|volvo|subaru|volkswagen|kia|mitsubishi|infiniti|lexus"
        r"|cadillac|dodge|mini|jaguar|land rover|bmw|mercedes|audi|porsche|tesla)"
        r"[a-zé\\-\\s]*\\b", text.lower()
    )
    INVALID_SUFFIXES = {"units", "inventory", "vehicles", "images", "stock"}

    cleaned = []
    for d in dealer_matches:
        d_clean = d.strip()
        if d_clean in DEALER_BLOCKLIST:
            continue
        parts = d_clean.split()
        if parts and parts[-1] not in INVALID_SUFFIXES:
            cleaned.append(d_clean)

    return list(set(cleaned))

def extract_syndicators(text):
    candidates = [
        "vauto", "easydeal", "car media", "icc", "homenet", "serti",
        "evolutionautomobiles", "spincar", "trader", "pbs", "google", "omni"
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
        "line_count": text.count("\\n") + 1
    }
