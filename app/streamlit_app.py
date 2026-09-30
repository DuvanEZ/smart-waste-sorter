"""Smart Waste Sorter - interactive web application (Streamlit).

Start it from the project folder with
    streamlit run app/streamlit_app.py
or double-click run_app.bat (Windows) / run ./run_app.sh (macOS, Linux).

The user gives one or more photos of a waste item (upload, webcam, sample image,
several files or a whole folder). Every input is validated, the trained model
predicts the material (cardboard, glass, metal, paper, plastic or trash), and the
app reports the class, the confidence, the alternatives and where to dispose of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from predictor import DEFAULT_MODEL_DIR, WasteClassifier  # noqa: E402
from recycling_guide import GENERAL_NOTE, advice_for  # noqa: E402
from validation import (ALLOWED_EXTENSIONS, MAX_FILE_MB, MAX_FOLDER_IMAGES, list_folder_images,  # noqa: E402
                        validate_image_bytes, validate_image_file)

SAMPLE_DIR = APP_DIR.parent / "sample_images"
UPLOAD_TYPES = sorted(e.lstrip(".") for e in ALLOWED_EXTENSIONS)

st.set_page_config(page_title="Smart Waste Sorter", page_icon="♻️", layout="wide")


# --------------------------------------------------------------------------------------
# Model (loaded once and cached for the whole session)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model ...")
def load_model() -> WasteClassifier:
    return WasteClassifier(DEFAULT_MODEL_DIR)


try:
    model = load_model()
except Exception as exc:  # missing/corrupted model files -> clear instructions instead of a crash
    st.error(f"The model could not be loaded: {exc}")
    st.info("Make sure the folder 'model' (with waste_classifier.onnx and model_info.json) is next to the "
            "'app' folder, and that the requirements are installed: pip install -r requirements.txt")
    st.stop()


# --------------------------------------------------------------------------------------
# Sidebar: settings and model summary
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.title("♻️ Smart Waste Sorter")
    st.write("Identifies the material of a waste item from a photo and tells you how to dispose of it.")
    threshold = st.slider(
        "Confidence threshold", min_value=0.30, max_value=0.95, value=float(model.default_threshold), step=0.05,
        help="Predictions below this probability are flagged as UNCERTAIN (check the item manually).")
    show_all = st.checkbox("Show the probability of every class", value=True)
    st.divider()
    st.subheader("Model")
    st.write(f"**{model.info.get('model_name', 'EfficientNet-B0')}**")
    if model.test_accuracy is not None:
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{model.test_accuracy:.1%}")
        c2.metric("Macro-F1", f"{model.test_macro_f1:.3f}")
    st.caption("Classes: " + ", ".join(model.class_names))
    st.caption(GENERAL_NOTE)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def probability_table(prediction) -> pd.DataFrame:
    rows = prediction.top_k if show_all else prediction.top_k[:3]
    return pd.DataFrame({"Class": [c for c, _ in rows], "Probability": [p for _, p in rows]})


def show_result(name: str, validation, prediction) -> None:
    """Image on the left, prediction card on the right."""
    left, right = st.columns([1, 1.3])
    with left:
        st.image(validation.image, caption=f"{name} ({validation.width}x{validation.height} px)", width=360)
    with right:
        guide = advice_for(prediction.label)
        st.markdown(f"### {guide['icon']} {prediction.label.upper()}")
        st.progress(min(max(prediction.confidence, 0.0), 1.0), text=f"Confidence: {prediction.confidence:.1%}")
        if prediction.is_confident:
            st.success(f"Confident prediction (≥ {threshold:.0%}).")
        else:
            second = prediction.top_k[1]
            st.warning(f"UNCERTAIN: confidence below {threshold:.0%}. It could also be **{second[0]}** "
                       f"({second[1]:.1%}). Try a clearer photo of a single item on a plain background.")
        for w in validation.warnings:
            st.info(w)
        st.markdown(f"**Where does it go?** {guide['bin']}")
        for tip in guide["tips"]:
            st.markdown(f"- {tip}")
        st.dataframe(
            probability_table(prediction), hide_index=True,
            column_config={"Probability": st.column_config.ProgressColumn(
                "Probability", format="%.3f", min_value=0.0, max_value=1.0)})


def classify_many(items):
    """items: list of (name, validation). Returns a results DataFrame."""
    valid = [(n, v) for n, v in items if v.ok]
    predictions = model.predict_batch([v.image for _, v in valid], threshold) if valid else []
    by_name = {n: p for (n, _), p in zip(valid, predictions)}
    rows = []
    for name, v in items:
        if not v.ok:
            rows.append({"File": name, "Prediction": "-", "Confidence": None, "Second choice": "-",
                         "Status": "REJECTED", "Disposal": "-", "Message": v.error})
            continue
        p = by_name[name]
        rows.append({"File": name, "Prediction": p.label, "Confidence": p.confidence,
                     "Second choice": f"{p.top_k[1][0]} ({p.top_k[1][1]:.1%})",
                     "Status": "OK" if p.is_confident else "UNCERTAIN",
                     "Disposal": advice_for(p.label)["bin"], "Message": " ".join(v.warnings)})
    return pd.DataFrame(rows)


def show_batch(results: pd.DataFrame) -> None:
    n_ok = int((results["Status"] == "OK").sum())
    n_unc = int((results["Status"] == "UNCERTAIN").sum())
    n_rej = int((results["Status"] == "REJECTED").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files", len(results))
    c2.metric("Confident", n_ok)
    c3.metric("Uncertain", n_unc)
    c4.metric("Rejected", n_rej)
    counts = results.loc[results["Status"] != "REJECTED", "Prediction"].value_counts()
    if len(counts):
        st.bar_chart(counts.reindex(model.class_names).fillna(0))
    st.dataframe(results, hide_index=True, column_config={
        "Confidence": st.column_config.ProgressColumn("Confidence", format="%.3f", min_value=0.0, max_value=1.0)})
    st.download_button("⬇️ Download results as CSV", results.to_csv(index=False).encode("utf-8"),
                       file_name="waste_predictions.csv", mime="text/csv")


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
st.title("♻️ Smart Waste Sorter")
st.caption("Deep-learning classifier for recyclable waste: cardboard · glass · metal · paper · plastic · trash")
tab_one, tab_batch, tab_model, tab_help = st.tabs(
    ["🔍 Classify an item", "🗂️ Batch classification", "📊 Model performance", "❓ Help"])

# ---- 1. single image --------------------------------------------------------------------
with tab_one:
    source = st.radio("How do you want to provide the image?",
                      ["Upload a photo", "Take a photo with the webcam", "Use a sample image"], horizontal=True)
    name, data = None, None
    if source == "Upload a photo":
        up = st.file_uploader(f"Choose an image ({', '.join(UPLOAD_TYPES)}; max {MAX_FILE_MB} MB)",
                              type=UPLOAD_TYPES, accept_multiple_files=False, key="single_upload")
        if up is not None:
            name, data = up.name, up.getvalue()
    elif source == "Take a photo with the webcam":
        shot = st.camera_input("Place ONE item in front of the camera and take a photo")
        if shot is not None:
            name, data = "webcam.jpg", shot.getvalue()
    else:
        samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS)
        if not samples:
            st.warning("No sample images found in the 'sample_images' folder.")
        else:
            choice = st.selectbox("Sample image", [p.name for p in samples])
            chosen = SAMPLE_DIR / choice
            name, data = chosen.name, chosen.read_bytes()

    if data is not None:
        validation = validate_image_bytes(name, data)
        if not validation.ok:
            st.error(f"❌ {validation.error}")
        else:
            with st.spinner("Classifying ..."):
                prediction = model.predict(validation.image, threshold)
            show_result(name, validation, prediction)
    else:
        st.info("Provide an image to get a prediction. Tip: photograph a single item, filling most of the frame.")

# ---- 2. several images -----------------------------------------------------------------
with tab_batch:
    st.write("Classify several images at once, then download the results as a CSV file.")
    mode = st.radio("Input", ["Upload several files", "Classify a folder on this computer"], horizontal=True)
    if mode == "Upload several files":
        ups = st.file_uploader("Choose images", type=UPLOAD_TYPES, accept_multiple_files=True, key="multi_upload")
        if ups and st.button("Classify uploaded images", type="primary"):
            with st.spinner(f"Classifying {len(ups)} images ..."):
                st.session_state["batch"] = classify_many([(u.name, validate_image_bytes(u.name, u.getvalue()))
                                                           for u in ups])
    else:
        folder = st.text_input("Folder path", placeholder=r"e.g. C:\Users\me\Pictures\waste  or  /Users/me/waste",
                               help=f"All supported images directly inside the folder are classified (max {MAX_FOLDER_IMAGES}).")
        if st.button("Classify folder", type="primary"):
            files, error = list_folder_images(folder)
            if error:
                st.error(f"❌ {error}")
            else:
                if len(files) == MAX_FOLDER_IMAGES:
                    st.warning(f"Only the first {MAX_FOLDER_IMAGES} images are classified.")
                with st.spinner(f"Classifying {len(files)} images ..."):
                    st.session_state["batch"] = classify_many([(f.name, validate_image_file(f)) for f in files])
    if "batch" in st.session_state:
        show_batch(st.session_state["batch"])

# ---- 3. model card ---------------------------------------------------------------------
with tab_model:
    info = model.info
    metrics = info.get("metrics") or {}
    st.subheader("How good is the model?")
    st.write("Results on the held-out **test set** (images never seen during training or model selection).")
    if metrics:
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        cols[1].metric("Macro-F1", f"{metrics.get('macro_f1', 0):.3f}")
        cols[2].metric("Top-2 accuracy", f"{metrics.get('top2_accuracy', 0):.1%}")
        cols[3].metric("ROC-AUC (macro)", f"{metrics.get('roc_auc_ovr_macro', 0):.3f}")
        if metrics.get("accuracy_ci"):
            lo, hi = metrics["accuracy_ci"]
            st.caption(f"95% bootstrap confidence interval of the accuracy: {lo:.1%} - {hi:.1%} "
                       f"(n = {metrics.get('n', '?')} test images).")
    if info.get("per_class"):
        st.dataframe(pd.DataFrame(info["per_class"]), hide_index=True)
    cm_path = DEFAULT_MODEL_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion matrix on the test set", width=520)
    st.subheader("About the model")
    st.write(f"- Architecture: {info.get('model_name')}, input {info.get('input_size')}x{info.get('input_size')} px")
    st.write(f"- Training data: {info.get('dataset')}")
    st.write(f"- Model file: ONNX, {info.get('onnx_file_mb', '?')} MB; average CPU time "
             f"{(info.get('cpu_latency_ms') or {}).get('median_ms', float('nan')):.0f} ms per image")
    st.write("- Limitations: one item per photo; the model only knows these six materials, so other objects "
             "(e.g. food, electronics, textiles) are forced into the closest class - watch the confidence.")

# ---- 4. help ---------------------------------------------------------------------------
with tab_help:
    st.subheader("Quick guide")
    st.markdown(f"""
1. **Classify an item** - upload a photo, take one with the webcam, or pick a sample image.
   The app shows the predicted material, the confidence, the other likely classes and the correct bin.
2. **Batch classification** - upload several photos or type the path of a folder. The results table can be
   downloaded as CSV (columns: File, Prediction, Confidence, Second choice, Status, Disposal, Message).
3. **Confidence threshold** (sidebar) - predictions below it are marked **UNCERTAIN**.

**Accepted input:** {', '.join(UPLOAD_TYPES)} files up to {MAX_FILE_MB} MB, at least 32x32 px.
Invalid files (wrong type, corrupted, empty, too small, extreme shape) are rejected with a message;
unusual but usable files (grey-scale, transparent, low resolution) are accepted with a warning.

**Meaning of the output:** *Prediction* = most likely material; *Confidence* = calibrated probability (0-1);
*Status* = OK (confidence ≥ threshold), UNCERTAIN (below threshold) or REJECTED (invalid input).

**Best results:** one object, well lit, filling most of the picture, plain background.
""")
