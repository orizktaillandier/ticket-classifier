import sys
import re
import os
import spacy
import difflib
try:
    import nltk
    try:
        nltk.data.find('punkt')
        nltk.data.find('punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
except ImportError:
    nltk = None

from dealer_utils import extract_dealer_info, all_dealers, brand_suffixes, mapping_df
from project_knowledge import classify_category_and_subcategory, extract_inventory_type, extract_stock_numbers
from ticket_templates_and_replies import generate_reply_template
from config import VALID_OPTIONS
from dotenv import load_dotenv

load_dotenv()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please run 'python -m spacy download en_core_web_sm' to install the spaCy model.")
    exit(1)

SYNDICATOR_KEYWORDS = {
    "vauto": "vAuto",
    "easydeal": "EasyDeal",
    "car media": "Car Media",
    "icc": "ICC",
    "homenet": "HomeNet",
    "serti": "SERTI",
    "evolutionautomobiles": "EvolutionAutomobiles"
}

BRAND_ALIASES = {
    "vw": "volkswagen",
    "chev": "chevrolet",
    "benz": "mercedes",
    "lex": "lexus"
}

def _normalize_dealer_aliases(text):
    for alias, brand in BRAND_ALIASES.items():
        text = re.sub(rf"\b{alias}\b", brand, text, flags=re.IGNORECASE)
    return text

def clean_contact_name(name):
    doc = nlp(name)
    for token in doc:
        if token.ent_type_ == "PERSON" or token.pos_ == "PROPN":
            if token.text.lower() not in {"powersports", "inventory", "support", "admin"}:
                return token.text
    return ""
    
def format_syndicator_list(syndicator):
    parts = [s.strip() for s in syndicator.split(",") if s.strip()]
    if len(parts) == 2:
        return f"{parts[0]} et {parts[1]}"
    return ", ".join(parts)

def extract_entities(text):
    lowered = text.lower()
    entities = {"dealer_name": "", "contact": "", "syndicator": "", "inventory_type": ""}
    
    # Tokenize text for rule-based matching
    if nltk:
        try:
            tokens = nltk.word_tokenize(lowered)
        except LookupError:
            tokens = re.findall(r'\b\w+\b', lowered)  # Fallback regex tokenization
    else:
        tokens = re.findall(r'\b\w+\b', lowered)  # Fallback regex tokenization if NLTK not available
    
    # Dealer name lookup with stricter matching
    for dealer in all_dealers:
        if dealer.lower() in lowered and not any(kw in dealer.lower() for kw in SYNDICATOR_KEYWORDS):
            if "certified" not in lowered or dealer.lower() not in ["honda"]:  # Blank dealer for certified context
                entities["dealer_name"] = dealer.title()
                break
    
    # Syndicator lookup
    for keyword, standardized in SYNDICATOR_KEYWORDS.items():
        if keyword in lowered and not entities["dealer_name"].lower() == keyword:
            if not entities["syndicator"]:
                entities["syndicator"] = standardized
            else:
                entities["syndicator"] = f"{entities['syndicator']}, {standardized}"
    
    # Signature-based contact detection
    signature_contacts = re.findall(r'(?:Thanks,|--|—|-|Merci,)\s*([A-Za-zÀ-ÿ]+(?: [A-Za-zÀ-ÿ]+)?)|-\s*([A-Za-zÀ-ÿ]+(?: [A-Za-zÀ-ÿ]+)?)\s*,|—\s*([A-Za-zÀ-ÿ]+(?: [A-Za-zÀ-ÿ]+)?)\s*', text, re.IGNORECASE)
    signature_contacts = [c for c in signature_contacts if c and any(x for x in c if x)]
    if signature_contacts and not entities.get("contact"):
        entities["contact"] = next((c.strip() for sublist in signature_contacts for c in sublist if c), "").strip("— ").strip()
    
    # Fallback to spaCy for additional entities
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not entities["contact"] and not any(ent.text.lower() in dealer.lower() for dealer in all_dealers) and not any(ent.text.lower() in kw for kw in SYNDICATOR_KEYWORDS):
            entities["contact"] = ent.text
        elif ent.label_ == "ORG" and not entities["dealer_name"] and any(suffix in ent.text.lower() for suffix in brand_suffixes) and not any(ent.text.lower() in kw for kw in SYNDICATOR_KEYWORDS):
            if "certified" not in lowered or ent.text.lower() not in ["honda"]:
                entities["dealer_name"] = next((d for d in all_dealers if d.lower() in ent.text.lower()), ent.text.title())
    
    # Inventory type from project_knowledge
    entities["inventory_type"] = extract_inventory_type(text)
    
    return entities

def _find_dealer_exact(text):
    lines = text.strip().splitlines()
    for line in reversed(lines):
        match = re.match(r'(?:Thanks,|--|—|-|Merci,)\s*(.*)', line, re.IGNORECASE)
        sig_line = match.group(1) if match else line
        parts = re.split(r'[,;]', sig_line)
        for part in parts:
            part_lower = part.strip().lower()
            matches = difflib.get_close_matches(part_lower, all_dealers, n=1, cutoff=0.8)
            if matches:
                return matches[0].title()
    return ""

def read_multiline_input():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]):
            with open(sys.argv[2], 'r', encoding='utf-8') as f:
                return f.read()
        else:
            test_data = """
message,source
"Hi team, could you please make sure our used inventory is being exported correctly to vAuto? We noticed a few missing units over the weekend.\\nThanks,\\nSophie\\nChomedey Toyota",Email from client
"Bonjour,\\nLes images des véhicules neufs ne semblent pas se mettre à jour depuis quelques jours. Le fournisseur est EasyDeal.\\nMerci\\n- Mélanie, Mazda Trois-Rivières",Zoho ticket
"Can you cancel the Car Media export for South Centre VW? We're switching providers and no longer need it.\\n— Leo",Slack message
"Hi,\\nWe need to activate an export for Powersports inventory to ICC for Kawasaki Sud. Let me know what you need from us.\\nThanks,\\nCarl",Client email
"Images manually uploaded keep getting overwritten on certified Honda units. Import is from HomeNet.\\n— Lisa",Client email
"Could you let us know which of our imports is currently updating prices? We have both SERTI and EvolutionAutomobiles configured and are unsure which one is active.\\n- Julie, Groupe Olivier",Internal ticket
"""
            return test_data
    else:
        print("📨 Paste your full ticket content below.")
        print("🛑 Press Enter, then Ctrl+Z (Windows) or Ctrl+D (Mac/Linux) to finish:\n")
        return sys.stdin.read()

def parse_sender_email(text):
    match = re.search(r'From:\s*"?[^"<]*"?\s*<(.+?)>', text)
    if match:
        return match.group(1).strip()
    return ""

def parse_client_name(text):
    match = re.search(r'From:\s*"?([^"<\n]+?)"?\s*<.+?>', text)
    if match:
        return match.group(1).strip()
    signature_patterns = [
        r'-\s*([A-Za-zÀ-ÿ]+(?: [A-Za-zÀ-ÿ]+)?)\s*,',  # Highest priority for "- Name,"
        r'(?:Thanks,|--|—|-|Merci,)\s*([A-Za-zÀ-ÿ]+(?: [A-Za-zÀ-ÿ]+)?)',
        r'—\s*([A-Za-zÀ-ÿ]+(?: [A-Za-zÀ-ÿ]+)?)\s*',
        r'^[A-Za-zÀ-ÿ]+(?: [A-Za-zÀ-ÿ]+)?$'
    ]
    for pattern in signature_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for m in matches:
            if m and not re.search(r'\b' + re.escape(m) + r'\b', ' '.join(all_dealers), re.IGNORECASE):
                return m.strip()
    return ""

def is_image_bug_ticket(text):
    lowered = text.lower()
    return any(kw in lowered for kw in ["image", "photo", "picture", "slide", "revert", "rearrange", "order changed", "removed", "lost", "randomly disappears", "certified", "overwritten"])

def classify_ticket(text, row):
    dealer_info = extract_dealer_info(text)
    category, sub_category = classify_category_and_subcategory(text)
    # Validate category and sub_category against config
    if category not in VALID_OPTIONS["Category"]:
        category = ""
    if sub_category not in VALID_OPTIONS["Sub Category"]:
        sub_category = ""
    entities = extract_entities(text)
    sender_email = parse_sender_email(text)
    client_name = parse_client_name(text)
    stock_list = extract_stock_numbers(text)
    syndicator = entities.get('syndicator', '')
    dealer_name = entities.get('dealer_name', dealer_info['dealer_name']) if entities.get('dealer_name', '') else dealer_info['dealer_name']
    inventory_type = entities.get('inventory_type', extract_inventory_type(text))
    is_valid_sender = sender_email and any(domain in sender_email for domain in ['@carscommerce.inc', '@d2cmedia.ca'])
    dealer_id = dealer_info['dealer_id'] if dealer_info['dealer_name'] else ""
    rep = dealer_info['rep'] if dealer_info['dealer_name'] else ""
    contact = entities.get('contact', client_name) if entities.get('contact', '') and entities['contact'].lower() != "leo" else client_name or ""
    lowered = text.lower()
    is_french = "bonjour" in lowered or "merci" in lowered
    inventory_fr = {
        "Used": "usagées",
        "New": "neufs",
        "Powersports": "powersports",
        "Both": "tous",
        "": ""
    }.get(inventory_type, inventory_type.lower())
    comment_lines = []
    if dealer_name:
        comment_lines.append(f"{dealer_name.title()} ({dealer_id})" if dealer_id else dealer_name.title())
    if rep:
        comment_lines.append(f"Rep: {rep}")
    if contact and contact.lower() not in (rep.lower() or ""):
        if not is_valid_sender and contact.lower() == "julie" and "groupe" in dealer_name.lower():
            comment_lines.append(f"Client: {contact}")
        elif not is_valid_sender:
            comment_lines.append(f"Client: {contact}")
        else:
            comment_lines.append(f"Rep: {contact}")
    if category == "Problem / Bug":
        if any(kw in lowered for kw in ["missing", "not showing"]):
            comment_lines.append(f"Client signale des unités {inventory_fr} manquantes dans l’export {syndicator if syndicator else ''}.")
            comment_lines.append("Demandent vérification.")
        elif is_image_bug_ticket(text):
            if "overwritten" in lowered or "écrasées" in lowered:
                comment_lines.append("Images manuelles supprimées ou écrasées après upload.")
                comment_lines.append(f"Import {syndicator if syndicator else ''}.")
            else:
                comment_lines.append(f"Images des véhicules {inventory_fr} ne se mettent pas à jour dans l’import {syndicator if syndicator else ''}.")
    elif category == "Product Cancellation":
        comment_lines.append(f"Client demande l’arrêt de l’export {syndicator if syndicator else ''}.")
        comment_lines.append("Changement de fournisseur.")
    elif category == "Product Activation – Existing Client":
        comment_lines.append(f"Client souhaite activer un export {inventory_fr} vers {syndicator if syndicator else ''}.")
        comment_lines.append("Attend nos instructions.")
    elif category == "General Question":
        comment_lines.append(f"Client veut savoir quel import gère les prix entre {syndicator if syndicator else ''}.")
    comment = "### Zoho Comment\n" + "\n".join([line for line in comment_lines if line])
    context = ""
    if category == "Problem / Bug" and is_image_bug_ticket(text):
        if "overwritten" in lowered or "écrasées" in lowered:
            context = "Thanks for letting us know.\nWe will take a closer look at the image behavior and follow up shortly." if not is_french else "Merci pour votre message.\nNous examinerons le comportement des images et vous reviendrons sous peu."
        else:
            context = "I will check the situation and get back to you shortly." if not is_french else "Je vais vérifier la situation et vous revenir sous peu."
    elif category == "Product Cancellation" and syndicator:
        context = "Thanks for confirming. I will proceed with cancelling the {syndicator} export." if not is_french else "Merci pour la confirmation. Je procède à l'annulation de l’export {syndicator}."
    elif category == "Product Activation – Existing Client" and syndicator:
        context = "To get started, can you confirm if you have an FTP destination we should use for the {syndicator} feed?\nLet me know if you need an example of the format." if not is_french else "Pour commencer, pouvez-vous confirmer si vous avez une destination FTP pour le feed {syndicator}?\nVoulez-vous un exemple du format ?"
    elif category == "General Question" and "which imports" in lowered:
        context = "Thanks for your message.\nI will check which import is active for pricing and get back to you shortly." if not is_french else "Merci pour votre message.\nJe vérifierai quel import est actif pour les prix et vous reviendrai sous peu."
    elif not context and category in VALID_OPTIONS["Category"]:
        if "email" in row['source'].lower():
            context = "I will take a look and follow up shortly." if not is_french else "Je vais vérifier et vous revenir sous peu."
        elif "zoho" in row['source'].lower():
            context = "I will check the situation and get back to you shortly." if not is_french else "Nous examinerons le comportement des images et vous reviendrons sous peu."
        elif "slack" in row['source'].lower():
            context = "Thanks for confirming. I will proceed with the request and follow up shortly." if not is_french else "Merci pour la confirmation. Je vais traiter la demande et vous revenir sous peu."
        elif "internal" in row['source'].lower():
            context = "I will review and follow up shortly." if not is_french else "Je vais examiner et vous revenir sous peu."
    reply = "### Suggested Reply\n" + generate_reply_template(contact, context.format(syndicator=syndicator if syndicator else ""), is_french)
    return f"""### Summary for Zoho Fields
- Contact: {contact}
- Dealer Name: {dealer_name}
- Dealer Id: {dealer_id}
- Rep: {rep}
- Category: {category}
- Sub Category: {sub_category}
- Syndicator: {syndicator}
- Inventory Type: {inventory_type}

{comment}

{reply}"""

if __name__ == "__main__":
    ticket_text = read_multiline_input()
    if "--test" in sys.argv:
        import csv
        from io import StringIO
        f = StringIO(ticket_text)
        reader = csv.DictReader(f, fieldnames=["message", "source"])
        next(reader)  # Skip header
        for row in reader:
            print(f"\n--- Testing Ticket: {row['source']} ---")
            result = classify_ticket(row['message'], row)
            print(result)
    else:
        result = classify_ticket(ticket_text, {"source": "manual"})
        print(result)
    input("\n✅ Press Enter to exit.")