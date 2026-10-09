"""Streamlit frontend.  Run:  streamlit run app.py"""
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

from src import db
from src.predictor import load_model, predict

st.set_page_config(page_title="AI-Based Fake Review Detection", page_icon="🕵️", layout="centered")

st.title("🕵️ AI-Based Fake Review Detection")
st.caption("NLP + TF-IDF + Machine Learning. Paste a product review and see how the model scores it.")

try:
    load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("💡 You can train the model now with the sample dataset by clicking the button below:")
    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Training model on dataset..."):
            from src.train import main as train_model
            train_model()
            st.success("Model trained successfully! Reloading...")
            st.rerun()
    st.stop()

EXAMPLES = {
    "— choose an example —": "",
    "Detailed, personal": "I bought this blender three weeks ago. The motor is loud but crushes ice fine, and the lid fits tightly. Cleaning takes a minute.",
    "Generic praise": "Great product. Highly recommend. Five stars. Very good quality and fast delivery.",
    "Very short": "Nice.",
}
choice = st.selectbox("Try an example (optional)", list(EXAMPLES))
review = st.text_area("Review", value=EXAMPLES[choice], height=160, placeholder="Enter a product review…")

if st.button("Analyze Review", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        result = predict(review)
        db.log_prediction(review, result["label"], result["confidence"])

        if result["is_fake"]:
            st.error(f"**Prediction:** {result['label']}")
        else:
            st.success(f"**Prediction:** {result['label']}")
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        st.progress(result["p_fake"], text=f"Probability fake: {result['p_fake'] * 100:.1f}%")

        st.subheader("Why?")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Words pushing toward Fake**")
            for w, s in result["fake_words"] or []:
                st.write(f"- `{w}` ({s:+.3f})")
            if not result["fake_words"]:
                st.write("_none_")
        with c2:
            st.markdown("**Words pushing toward Genuine**")
            for w, s in result["genuine_words"] or []:
                st.write(f"- `{w}` ({s:+.3f})")
            if not result["genuine_words"]:
                st.write("_none_")
        st.caption(
            "Explanation shows each word's contribution inside this model only. "
            "It shows what the model relies on, not proof a review is truly fake."
        )

with st.expander("Recent predictions (from local database)"):
    rows = db.recent_predictions(20)
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["time (UTC)", "review", "prediction", "confidence"]),
                     width="stretch")
    else:
        st.write("No predictions yet.")

st.divider()
st.caption("Educational project. A model trained on one dataset may not generalize to other websites.")

if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as _st_exists
        _in_streamlit = _st_exists()
    except Exception:
        _in_streamlit = False

    if not _in_streamlit:
        import sys
        try:
            from streamlit.web import cli as stcli
        except ImportError:
            import streamlit.cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
