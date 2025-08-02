import requests
from datetime import datetime
import streamlit as st
from llm_classifier import classify_ticket
import json

st.set_page_config(page_title="Ticket AI Classifier", layout="wide")

# Hide Streamlit footer and menu
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border-radius: 6px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
        color: white;
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
    ticket_input = st.text_area("Paste full email or ticket content here:", height=180)
    classify = st.button("🚀 Classify Ticket")

# Classification Section
if classify:
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
                    st.markdown(f"""
**Dealer Name**: `{zf.get("dealer_name", "")}`  
**Dealer ID**: `{zf.get("dealer_id", "")}`  
**Rep**: `{zf.get("rep", "")}`  
**Contact**: `{zf.get("contact", "")}`  
**Category**: `{zf.get("category", "")}`  
**Sub Category**: `{zf.get("sub_category", "")}`  
**Syndicator**: `{zf.get("syndicator", "")}`  
**Inventory Type**: `{zf.get("inventory_type", "")}`
""")

                    if edge:
                        st.warning(f"⚠️ Detected Edge Case: `{edge}`")

                    st.markdown("---")
                    st.markdown("### 📬 Communication Timeline")

                    # Parse sender lines for summary
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
                    # Add feedback button (Google Forms only, no file logging)
                    if st.button("❌ This classification is incorrect"):
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

            except Exception as e:
                st.error("❌ An unexpected error occurred.")
                st.exception(e)

# No local log file or download code anymore — all feedback is now cloud-based!
