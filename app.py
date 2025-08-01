import streamlit as st
from llm_classifier import classify_ticket
import json

st.set_page_config(page_title="Ticket AI Classifier", layout="wide")

st.title("🎟️ Ticket AI Classifier")
st.markdown("""
This tool classifies Zoho Desk tickets using your custom LLM pipeline.

- Paste an email/ticket below to extract Zoho fields and generate the Zoho comment.
- Dealer ID, rep, syndicator, and comment logic are dynamic.
""")

with st.expander("ℹ️ How to Use"):
    st.markdown("""
    - Paste the full email body, including headers or signature if available.
    - The system will automatically extract Dealer Name, Contact, Syndicator, and more.
    - Output will follow the Zoho classification format and include the comment + reply.
    """)

ticket_input = st.text_area("Paste full email or ticket content here:", height=250)

if st.button("Classify Ticket"):
    if not ticket_input.strip():
        st.error("Please paste a ticket or message.")
    else:
        try:
            result = classify_ticket(ticket_input.strip())
            st.success("✅ Classification complete.")
            st.subheader("📋 Zoho Fields")
            st.json(result["zoho_fields"])

            st.subheader("📝 Zoho Comment")
            st.code(result["zoho_comment"], language="markdown")

            #st.subheader("✉️ Suggested Reply")
            #st.code(result["suggested_reply"], language="markdown")

        except Exception as e:
            st.error("❌ An unexpected error occurred.")
            st.exception(e)
