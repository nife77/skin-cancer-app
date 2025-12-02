# Skin Cancer Diagnosis Prediction

This project is a simple end-to-end implementation of a skin cancer diagnosis system using:

- **ResNet50 (Transfer Learning, TensorFlow/Keras)**
- **Binary classification: Benign vs Malignant**
- **Grad-CAM** for explainable AI (XAI)
- **Streamlit** for the web UI
- **SQLite** to log predictions

## Basic Structure

```text
skin_cancer_project/
├─ data/
│   ├─ train/
│   │   ├─ benign/
│   │   └─ malignant/
│   ├─ val/
│   │   ├─ benign/
│   │   └─ malignant/
├─ models/
│   └─ skin_cancer_resnet50.h5   # (created after training)
├─ app/
│   ├─ app.py
│   ├─ gradcam_utils.py
│   └─ db_utils.py
├─ train_model.py
├─ requirements.txt
└─ README.md
```

## How to Use

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Prepare dataset**

Place your images in:

```text
data/train/benign/
data/train/malignant/
data/val/benign/
data/val/malignant/
```

3. **Train the model**

```bash
python train_model.py
```

This will create `models/skin_cancer_resnet50.h5`.

4. **Run the Streamlit app**

From the project root:

```bash
streamlit run streamlit_app.py
```

Then open the URL Streamlit shows (usually `http://localhost:8501`).

You can upload a skin lesion image, get a **Benign/Malignant** prediction, see a **Grad-CAM heatmap**, and save predictions into a local **SQLite** database.
