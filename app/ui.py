"""UI helpers kept as a module inside the `app` package.

This mirrors the Streamlit entrypoint but lives inside the package so the
root-level Streamlit script can import package resources without shadowing.
"""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

from app.db_utils import init_db, insert_prediction, get_all_predictions
from app.gradcam_utils import get_gradcam_heatmap, overlay_heatmap_on_image

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "skin_cancer_resnet50.h5"


st.set_page_config(page_title="Skin Cancer Classifier", layout="wide")


@st.cache_resource
def load_model(path: str):
    """Load and return a Keras model (cached as a resource)."""
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    if not Path(path).exists():
        st.error(f"Model file not found: {path}")
        return None
    try:
        model = load_model(path)
        return model
    except Exception as ex:
        st.exception(ex)
        return None


def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def pretty_class(prob: float) -> str:
    return f"{prob*100:.1f}% confidence"


def main():
    init_db()

    st.title("Skin Lesion classifier — demo")

    left, right = st.columns([2, 1])

    with left:
        uploaded = st.file_uploader("Upload a lesion image (jpg/png)", type=["jpg", "jpeg", "png"]) 
        target_size = (224, 224)

        model = load_model(MODEL_PATH)

        if uploaded is not None and model is not None:
            img = Image.open(BytesIO(uploaded.read()))
            st.subheader("Original image")
            st.image(img, use_column_width=True)

            input_arr = preprocess_image(img, target_size)

            # run prediction
            preds = model.predict(input_arr)
            # assuming binary classifier [benign, malignant] or single output
            if preds.shape[-1] == 1:
                prob = float(preds[0][0])
                cls_idx = int(prob >= 0.5)
                prob = prob if cls_idx == 1 else 1.0 - prob
                classes = ["benign", "malignant"] if cls_idx in (0,1) else ["unknown"]
            else:
                cls_idx = int(np.argmax(preds[0]))
                prob = float(np.max(preds[0]))
                classes = ["benign", "malignant"] if preds.shape[-1] == 2 else [f"class_{i}" for i in range(preds.shape[-1])]

            predicted_label = classes[cls_idx] if cls_idx < len(classes) else f"class_{cls_idx}"

            st.success(f"Prediction: {predicted_label} — {pretty_class(prob)}")

            # Grad-CAM
            heatmap, stats = get_gradcam_heatmap(model, input_arr)
            overlay = overlay_heatmap_on_image(img, heatmap)

            st.subheader("Grad-CAM overlay")
            st.image(overlay, use_column_width=True)

            # Save prediction to DB
            insert_prediction(Path(uploaded.name).name, predicted_label, float(prob))

    with right:
        st.header("Prediction history")
        rows = get_all_predictions()
        if rows:
            # simple display
            for r in rows[:20]:
                st.write({"id": r[0], "image": r[1], "pred": r[2], "confidence": r[3], "time": r[4]})
        else:
            st.info("No predictions recorded yet.")

    st.markdown("---")
    st.markdown("**Notes:** This demo expects a Keras .h5 model saved to `models/skin_cancer_resnet50.h5`. The repo already contains this file but the model may require TensorFlow 2.x to work correctly.")


if __name__ == "__main__":
    main()
