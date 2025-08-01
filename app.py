import os
import streamlit as st
from llm_classifier import classify_ticket

# Suppress Streamlit config warnings and email prompt
os.environ["STREAMLIT_SUPPRESS_CONFIG_WARNINGS"] = "true"
os.environ["STREAMLIT_EMAIL"] = ""

st.set_page_config(page_title="Ticket Classifier", layout="centered")

st.title("🎫 Zoho Ticket Classifier")
st.markdown("Paste the full message from a Zoho ticket below. The tool will auto-classify and generate a reply.")

ticket_input = st.text_area("📥 Paste Ticket Message", height=300)

if st.button("Classify Ticket"):
    if not ticket_input.strip():
        st.warning("Please paste a ticket message first.")
    else:
        with st.spinner("Classifying..."):
            result = classify_ticket(ticket_input.strip())
            fields = result.get("zoho_fields", {})

        st.subheader("📋 Zoho Fields")
        for label, key in [
            ("Contact", "contact"),
            ("Dealer Name", "dealer_name"),
            ("Dealer ID", "dealer_id"),
            ("Rep", "rep"),
            ("Category", "category"),
            ("Sub Category", "sub_category"),
            ("Syndicator", "syndicator"),
            ("Inventory Type", "inventory_type"),
        ]:
            st.text_input(label, value=fields.get(key, ""), disabled=True)

        st.subheader("📝 Zoho Comment")
        st.code(result.get("zoho_comment", ""), language="markdown")

        #st.subheader("✉️ Suggested Reply")
        #st.code(result.get("suggested_reply", ""), language="markdown")

        edge_case = result.get("edge_case", "")
        if edge_case:
            st.warning(f"⚠️ Edge Case Detected: {edge_case}")
