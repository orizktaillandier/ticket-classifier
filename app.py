
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
                st.subheader("🧾 Ticket Summary")
                st.markdown(f"""
                **Dealer Name**: {zf.get("dealer_name", "")}  
                **Dealer ID**: {zf.get("dealer_id", "")}  
                **Rep**: {zf.get("rep", "")}  
                **Category**: {zf.get("category", "")}  
                **Syndicator**: {zf.get("syndicator", "")}
                """)

                if edge:
                    st.warning(f"⚠️ Detected Edge Case: `{edge}`")

                # Keep existing expanders
                with st.expander("📋 Zoho Fields", expanded=False):
                    st.json(zf)

                with st.expander("📝 Zoho Comment", expanded=True):
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
