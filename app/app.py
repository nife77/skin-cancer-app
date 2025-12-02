# app/app.py
import os
import sys
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# add local app folder to path
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from gradcam_utils import get_gradcam_heatmap, overlay_heatmap_on_image, find_last_conv_layer
from db_utils import init_db, insert_prediction, get_all_predictions

st.set_page_config(layout='centered', page_title='Skin Cancer Diagnosis - XAI')

init_db()

@st.cache_resource
def load_model():
    # try common model paths
    possible = [
        os.path.join(os.path.dirname(CURRENT_DIR), 'models', 'skin_cancer_resnet50.h5'),
        os.path.join(os.path.dirname(CURRENT_DIR), 'models', 'skin_cancer_resnet50.keras'),
        os.path.join(os.path.dirname(CURRENT_DIR), 'models', 'skin_cancer_resnet50')
    ]
    for p in possible:
        if os.path.exists(p):
            try:
                model = tf.keras.models.load_model(p)
                return model
            except Exception as e:
                st.error(f'Error loading model at {p}: {e}')
                return None
    st.warning('Model not found in models/ folder. Please train and place model at models/skin_cancer_resnet50.h5')
    return None

model = load_model()
CLASS_NAMES = ['Benign', 'Malignant']

st.title('Skin Cancer Diagnosis Prediction (with Grad-CAM)')
st.markdown('Upload an image. Predictions will be shown along with a Grad-CAM heatmap.')

uploaded_file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert('RGB')  # keep full size
    except Exception as e:
        st.error(f'Could not read uploaded image: {e}')
        original_image = None

    if original_image is not None:
        st.image(original_image, caption='Uploaded Image', use_column_width=True)

        # prepare a resized copy for model input
        model_input = original_image.resize((224, 224))
        img_array = np.array(model_input).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        if model is None:
            st.error('Model is not loaded. Place a trained model in models/ and reload.')
        else:
            # prediction
            preds = model.predict(img_batch)
            pred_index = int(np.argmax(preds[0]))
            pred_class = CLASS_NAMES[pred_index]
            confidence = float(np.max(preds[0]) * 100.0)

            st.markdown(f'### Prediction: **{pred_class}**')
            st.write(f'Confidence: `{confidence:.2f}%`')

            # choose last conv layer automatically (robust)
            try:
                last_conv = find_last_conv_layer(model)
            except Exception as e:
                last_conv = None
                st.warning(f'Could not find last conv layer automatically: {e}')

            # compute gradcam (use model_input shape for prediction but overlay on original_image)
            try:
                heatmap = get_gradcam_heatmap(model, img_batch, last_conv, pred_index)
                # overlay using original full-size image (avoid shape mismatch)
                overlay = overlay_heatmap_on_image(original_image, heatmap, alpha=0.4)
                st.markdown('#### Grad-CAM Explanation')
                st.image(overlay, use_column_width=True)
                # optional debug: show heatmap stats
                st.write(f'Grad-CAM heatmap stats -> min: {np.min(heatmap):.3f}, max: {np.max(heatmap):.3f}')
            except Exception as e:
                st.error(f'Grad-CAM Error: {e}')

            if st.button('Save Prediction'):
                insert_prediction(getattr(uploaded_file, 'name', 'uploaded_image'), pred_class, confidence)
                st.success('Saved to local DB')

st.markdown('---')
st.header('Prediction History')
if st.button('Show history'):
    try:
        rows = get_all_predictions()
        if not rows:
            st.write('No saved predictions.')
        else:
            for r in rows:
                _id, image_name, pred, conf, ts = r
                st.write(f'ID: {_id} | {image_name} | {pred} | {conf:.2f}% | {ts}')
    except Exception as e:
        st.error(f'Error reading DB: {e}')
