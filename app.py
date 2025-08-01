import streamlit as st
from llm_classifier import classify_ticket_llm

st.set_page_config(page_title="Ticket AI Classifier", layout="wide")
st.title("🧠 Ticket AI Classifier")

st.markdown(
    """
    This tool classifies Zoho Desk tickets using your custom LLM pipeline.
    - Paste an email/ticket below to extract Zoho fields and generate the Zoho comment.
    - Dealer ID, rep, syndicator, and comment logic are dynamic.
    """
)

with st.expander("ℹ️ How to Use"):
    st.write("""
    - Paste the full body of a ticket or email in the text box below.
    - Click 'Classify Ticket' to run it through the classifier.
    - You will see extracted Zoho fields, the internal comment, and any detected edge cases.
    """)

ticket_input = st.text_area("Paste full email or ticket content here:", height=300)

if st.button("Classify Ticket"):
    if not ticket_input.strip():
        st.warning("Please paste a ticket message before running the classifier.")
    else:
        try:
            with st.spinner("Running classification..."):
                result = classify_ticket_llm(ticket_input.strip())

            st.success("✅ Classification complete!")

            fields = result.get("zoho_fields", {})
            comment = result.get("zoho_comment", "")
            edge = result.get("edge_case", "")

            st.subheader("📋 Summary for Zoho Fields")
            st.json(fields)

            st.subheader("📝 Zoho Comment")
            st.code(comment, language="vbnet")

            if edge:
                st.warning(f"⚠️ Edge case detected: {edge}")

        except Exception as e:
            st.error("❌ An unexpected error occurred.")
            st.exception(e)

st.markdown("---")
st.caption("Internal prototype — no data is stored or transmitted.")
