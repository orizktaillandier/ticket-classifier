def generate_reply_template(contact_name: str, context="", is_french=False) -> str:
    if not contact_name:
        contact_name = "there"
    greeting = "Bonjour" if is_french else "Hi"
    thanks = "Merci" if is_french else "Thanks"
    if is_french:
        base = f"{greeting} {contact_name},\n\n{thanks} pour votre message.\n"
        if context:
            base += "\n".join(line for line in context.split("\n") if line.strip())
        else:
            base += "Je vais vérifier et vous revenir sous peu.\n"
        base += "\nN’hésitez pas si vous avez d’autres questions.\n\n{thanks},\nOlivier"
    else:
        base = f"{greeting} {contact_name},\n\n{thanks} for reaching out.\n"
        if context:
            base += "\n".join(line for line in context.split("\n") if line.strip())
        else:
            base += "I will take a look and follow up shortly.\n"
        base += "\nLet me know if there’s anything else.\n\n{thanks},\nOlivier"
    return base.replace("{thanks}", thanks)