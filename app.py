import streamlit as st
from llm_classifier import classify_ticket
import json

st.set_page_config(page_title="Ticket AI Classifier", layout="wide")

# Hide Streamlit footer and menu
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🎟️ Ticket AI Classifier")
st.markdown("""
This tool classifies Zoho Desk tickets using your custom LLM pipeline.

- Paste an email/ticket below to extract Zoho fields and generate the Zoho comment.
- Dealer ID, rep, syndicator, and comment logic are dynamic.
""")

with st.sidebar:
    st.header("📝 Ticket Input")
    ticket_input = st.text_area("Paste full email or ticket content here:", height=250)
    classify = st.button("Classify Ticket")

col1, col2 = st.columns((1, 2))

if classify:
    if not ticket_input.strip():
        col1.error("Please paste a ticket or message.")
    else:
        with st.spinner("Classifying…"):
            try:
                result = classify_ticket(ticket_input.strip())
                col1.success("✅ Classification complete.")

                col2.subheader("📋 Zoho Fields")
                col2.json(result["zoho_fields"])

                col2.subheader("📝 Zoho Comment")
                col2.code(result["zoho_comment"], language="markdown")

            except Exception as e:
                col1.error("❌ An unexpected error occurred.")
                col1.exception(e)
