import streamlit as st
from llm_classifier import classify_ticket
import pandas as pd

st.set_page_config(page_title="Ticket AI Classifier", layout="wide")
st.title("🧠 Ticket AI Classifier")

st.markdown(
    """
    This tool classifies Zoho Desk tickets.
    - Paste an email/ticket below to extract Zoho fields and generate the Zoho comment.
    """
)

with st.expander("ℹ️ Instructions"):
    st.write("""
    - Paste the full body of a ticket or email in the box below.
    - The model will extract fields, suggest a Zoho comment, and flag any edge cases.
    """)

ticket_input = st.text_area("Paste ticket or email here:", height=300)

if st.button("Classify Ticket") and ticket_input.strip():
    try:
        with st.spinner("Classifying ticket..."):
            result = classify_ticket(ticket_input.strip())
        st.success("✅ Classification complete!")

        fields = result.get("zoho_fields", {})
        zoho_comment = result.get("zoho_comment", "")
        edge_case = result.get("edge_case", "")

        st.subheader("📋 Summary for Zoho Fields")
        st.json(fields)

        st.subheader("📝 Zoho Comment")
        st.code(zoho_comment, language="vbnet")

        if edge_case:
            st.warning(f"⚠️ Edge case detected: {edge_case}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("Powered by your production logic – no static business rules.")

