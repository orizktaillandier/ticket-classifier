import requests
import subprocess
from datetime import datetime
import streamlit as st
from llm_classifier import classify_ticket
import json

st.set_page_config(page_title="Ticket AI Classifier", layout="wide")

# Custom CSS
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

# SIDEBAR: Input + Export Toggle
with st.sidebar:
    st.header("📝 Ticket Input")
    if "ticket_input" not in st.session_state:
        st.session_state.ticket_input = ""

    ticket_input = st.text_area(
        "Ticket or Email Content",
        value=st.session_state.ticket_input,
        placeholder="Paste the full ticket or email body here...",
        height=260
    )

    classify_col, clear_col = st.columns([1, 1])
    with classify_col:
        classify = st.button("🚀 Classify Ticket", use_container_width=True)
    with clear_col:
        if st.button("🧹 Clear Fields", use_container_width=True):
            st.session_state.ticket_input = ""
            st.experimental_rerun()

    # # Export Toggle Section
    # # st.markdown("---")
    # # st.subheader("⚙️ Export Toggle Tools")
    
    #export_action = st.selectbox("Select Export Action", ["Enable Export", "Disable Export"])
    #dealer_id = st.text_input("Dealer ID", placeholder="e.g. 128")
    #syndicator = st.text_input("Syndicator Name", placeholder="e.g. trader")
    #inventory_types = st.multiselect("Inventory Types", ["Usagé", "Neuf", "Démonstrateur"], default=["Neuf"])
    
    #if st.button("Run Export Script"):
       # if not dealer_id.strip() or not syndicator.strip() or not inventory_types:
            #st.error("Please fill out all export fields.")
        #else:
            #for inv_type in inventory_types:
                #st.markdown(f"**Running export for:** `{inv_type}`")
    
                #script = "export_toggle_enable.py" if export_action == "Enable Export" else "export_toggle_disable.py"
                #result = subprocess.run(
                    #["python3", script, dealer_id.strip(), syndicator.strip(), inv_type],
                    #capture_output=True, text=True
               # )
    
                #if result.returncode == 0:
                    #st.success(f"✅ {inv_type} script ran successfully.")
                #else:
                   # st.error(f"❌ {inv_type} script failed.")
                #st.code(result.stdout + "\n" + result.stderr)

# MAIN: Classifier Output
if classify:
    st.session_state.ticket_input = ticket_input
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
