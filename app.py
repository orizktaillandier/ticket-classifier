import requests
from datetime import datetime
import streamlit as st
from llm_classifier import classify_ticket
import json

st.set_page_config(page_title="Ticket AI Classifier", layout="wide")

# Custom CSS for better textarea and button
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border-radius: 8px;
        padding: 0.7em 1.4em;
        font-size: 1.08em;
        margin-top: 0.5em;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
        color: white;
    }
    textarea {
        font-size: 1.12em !important;
        padding: 1.1em !important;
        border-radius: 8px !important;
        border: 1.2px solid #bfc9d1 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎟️ Ticket AI Classifier")

st.markdown("""
This tool classifies Zoho Desk tickets using your custom LLM pipeline.

- Paste a full email or ticket below and click **Classify Ticket**.
- Dealer ID, rep, syndicator, and comment logic are dynamically detected.
""")

# Sidebar Input
with st.sidebar:
    st.header("📝 Ticket Input")

    if "ticket_input" not in st.session_state:
        st.session_state.ticket_input = ""

    ticket_input = st.text_area(
        "Ticket or Email Content",
        value=st.session_state.ticket_input,
        placeholder="Paste the full ticket or email body here...",
        height=260,
    )

     # Buttons: side-by-side and centered
    classify_col, clear_col = st.columns([1, 1])
    with classify_col:
        classify = st.button("🚀 Classify Ticket", use_container_width=True)
    with clear_col:
        if st.button("🧹 Clear Fields", use_container_width=True):
            st.session_state.ticket_input = ""
            ticket_input = ""
            classify = False

    # Always keep in sync
    st.session_state.ticket_input = ticket_input

# Classification Section
if classify:
    ticket_input = st.session_state.ticket_input
    if not ticket_input.strip():
        st.error("Please paste a ticket or message.")
    else:
        with st.spinner("Classifying…"):
            try:
                result = classify_ticket(ticket_input.strip())
                st.success("✅ Classification complete.")
                zf = result.get("zoho_fields", {})
                edge = result.get("edge_case", "")
                raw_text = ticket_input.strip()

                # Split layout: left (Zoho Fields + Timeline), right (Zoho Comment)
                left_col, right_col = st.columns([2, 1])

                with left_col:
                    st.markdown("### 🧾 Zoho Fields")
                
                    # Card-style container using Markdown + CSS
                    st.markdown("""
                    <div style='
                        background-color: #1e1e1e;
                        padding: 1.2em 1.5em;
                        border-radius: 12px;
                        border: 1px solid #444;
                        margin-bottom: 1em;
                        font-family: monospace;
                        font-size: 0.95em;
                    '>
                    <p><strong style='color:#ccc;'>Dealer Name:</strong> <code>{dealer_name}</code></p>
                    <p><strong style='color:#ccc;'>Dealer ID:</strong> <code>{dealer_id}</code></p>
                    <p><strong style='color:#ccc;'>Rep:</strong> <code>{rep}</code></p>
                    <p><strong style='color:#ccc;'>Contact:</strong> <code>{contact}</code></p>
                    <p><strong style='color:#ccc;'>Category:</strong> <code>{category}</code></p>
                    <p><strong style='color:#ccc;'>Sub Category:</strong> <code>{sub_category}</code></p>
                    <p><strong style='color:#ccc;'>Syndicator:</strong> <code>{syndicator}</code></p>
                    <p><strong style='color:#ccc;'>Inventory Type:</strong> <code>{inventory_type}</code></p>
                    </div>
                    """.format(
                        dealer_name=zf.get("dealer_name", ""),
                        dealer_id=zf.get("dealer_id", ""),
                        rep=zf.get("rep", ""),
                        contact=zf.get("contact", ""),
                        category=zf.get("category", ""),
                        sub_category=zf.get("sub_category", ""),
                        syndicator=zf.get("syndicator", ""),
                        inventory_type=zf.get("inventory_type", ""),
                    ), unsafe_allow_html=True)

                    feedback = st.button("❌ This classification is incorrect", key="flag_button_left_col")
                    if feedback:
                        log_entry = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "edge_case": edge,
                            "zoho_fields": json.dumps(zf),
                            "zoho_comment": result.get("zoho_comment", ""),
                            "input_text": ticket_input.strip()
                        }
                        form_url = "https://docs.google.com/forms/d/e/1FAIpQLSfIJgy3DdtSQsZN6G4asdZyiWaf2Qb-8_9fwQLxp74sFTMx4g/formResponse"
                        payload = {
                            "entry.2041497043": log_entry["timestamp"],
                            "entry.827201251": log_entry["edge_case"],
                            "entry.1216884505": log_entry["zoho_fields"],
                            "entry.1859746012": log_entry["zoho_comment"],
                            "entry.91556361": log_entry["input_text"]
                        }
                        r = requests.post(form_url, data=payload)
                        if r.status_code == 200:
                            st.success("📝 Feedback sent to Google Sheets! Thank you!")
                        else:
                            st.info("Feedback submitted. Check Google Sheets to confirm receipt.")

                    if edge:
                        st.warning(f"⚠️ Detected Edge Case: `{edge}`")

                    st.markdown("---")
                    st.markdown("### 📬 Communication Timeline")

                    timeline = []
                    lines = raw_text.splitlines()
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line.lower().startswith("from:") or "wrote:" in line.lower():
                            if i + 1 < len(lines):
                                preview = lines[i + 1].strip()
                                summary = f"- **{line}**\n  → _{preview[:100]}..._"
                                timeline.append(summary)
                    if not timeline:
                        preview = raw_text[:150].replace("\n", " ")
                        timeline = [f"**Message:** _{preview}..._"]

                    st.markdown("\n\n".join(timeline))

                with right_col:
                    st.markdown("### 📝 Zoho Comment")
                    st.code(result["zoho_comment"], language="markdown")
                    st.download_button(
                        label="📋 Copy Zoho Comment",
                        data=result["zoho_comment"],
                        file_name="zoho_comment.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error("❌ An unexpected error occurred.")
                st.exception(e)
