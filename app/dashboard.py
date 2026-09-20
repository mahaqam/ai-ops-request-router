import streamlit as st

from src.predict import load_model

st.set_page_config(page_title="AI Operations Request Router", layout="centered")
st.title("AI Operations Request Router")
st.write("Enter an internal service request to route it to one of the trained operational categories.")

model = load_model()
text = st.text_area("Request text", height=180)
if st.button("Route request", type="primary"):
    if not text.strip():
        st.warning("Enter a request first.")
    else:
        label = str(model.predict([text])[0])
        st.success(f"Predicted category: {label}")
