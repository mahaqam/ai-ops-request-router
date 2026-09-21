from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import load_model

EXAMPLES = {
    "Hardware": "My laptop screen keeps flickering and the docking station is no longer detected.",
    "Access": "I cannot access the finance shared drive after my account permissions changed.",
    "HR Support": "I need help updating my employee information and benefits details.",
    "Storage": "The shared project folder is out of storage space and uploads are failing.",
    "Purchase": "Please help me request a new monitor and keyboard for my workstation.",
}

st.set_page_config(
    page_title="AI Operations Request Router",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: rgba(127,127,127,.08); margin-bottom: 1rem;}
    .hero h1 {margin: 0 0 .35rem 0;}
    .muted {opacity: .78;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧭 AI Operations Request Router</h1>
      <div class="muted">An NLP prototype for routing internal service requests to operational teams.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
m1.metric("Training records", "47,837")
m2.metric("Operational categories", "8")
m3.metric("Held-out macro F1", "86.2%")

st.caption(
    "Benchmark result: Linear SVM with unigram/bigram TF-IDF on an 80/20 stratified split. This is an in-dataset portfolio benchmark, not a production SLA."
)

@st.cache_resource
def get_model():
    try:
        return load_model(), None
    except FileNotFoundError as exc:
        return None, str(exc)

model, model_error = get_model()

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Try the router")
    example_name = st.selectbox("Load an example", ["Choose an example"] + list(EXAMPLES))
    if "request_text" not in st.session_state:
        st.session_state.request_text = ""
    if example_name != "Choose an example":
        st.session_state.request_text = EXAMPLES[example_name]

    text = st.text_area(
        "Internal service request",
        key="request_text",
        height=230,
        placeholder="Example: I cannot access the finance shared drive after my permissions changed.",
    )
    run = st.button("Route request", type="primary", use_container_width=True, disabled=model is None)

    if model is None:
        st.warning(
            "The interactive model artifact is not present in this checkout yet. The repository is prepared for Streamlit deployment; "
            "generate models/request_router.joblib with src/train.py before publishing the live demo."
        )

with right:
    st.subheader("Routing result")
    if run and model is not None:
        if not text.strip():
            st.warning("Enter a request first.")
        else:
            label = str(model.predict([text])[0])
            st.success(label)
            st.caption("Predicted operational category")
    elif model is not None:
        st.info("Choose an example or type a request, then run the classifier.")
    else:
        st.info("Benchmark and project information remain visible while the deployment model artifact is being prepared.")

    st.markdown("**Supported categories**")
    st.write(
        "Hardware · HR Support · Access · Miscellaneous · Storage · Purchase · Internal Project · Administrative rights"
    )

st.divider()
st.subheader("What this project demonstrates")
col1, col2, col3 = st.columns(3)
col1.markdown("**Applied NLP**\n\nTF-IDF text features with class-weighted Logistic Regression and Linear SVM baselines.")
col2.markdown("**Model evaluation**\n\nAccuracy, macro/weighted F1, per-class metrics, and confusion analysis.")
col3.markdown("**Application layer**\n\nA Streamlit interface plus a separate FastAPI prediction endpoint in the repository.")

with st.expander("Scope & limitations"):
    st.write(
        "This is a portfolio prototype evaluated on the supplied dataset. Real internal traffic can contain new categories, multilingual text, "
        "sensitive data, abbreviations, and domain shift. A production router should add confidence/abstention logic, human fallback, monitoring, privacy controls, and retraining."
    )

st.link_button(
    "View source on GitHub",
    "https://github.com/mahaqam/ai-ops-request-router",
    use_container_width=True,
)
