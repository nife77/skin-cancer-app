**Explainable AI for Skin Cancer Detection**

This project develops an explainable AI application for the detection of skin cancer from dermatoscopic images. It moves beyond a simple "black box" model by providing a visual explanation for its predictions, thereby building trust and serving as a valuable tool for assisting in clinical decisions.

**Key Features**

Explainable AI (XAI): Implements Grad-CAM to generate heatmaps that show exactly which parts of an image the model used to make its prediction.

Accessible & User-Friendly: Built with Streamlit, the application features a clean, interactive web interface.

High Performance: Utilizes a powerful, pre-trained ResNet model with transfer learning to achieve high accuracy with a relatively small dataset.

Robust Architecture: The project is designed with a modular approach, separating data management, model logic, and the user interface for easy maintenance and future upgrades.

**🛠 Technologies Used**

Deep Learning: TensorFlow and Keras

Image Processing: scikit-image & Pillow

Web Framework: Streamlit

Data Management: SQLite

Version Control: Git & GitHub

Core Model: ResNet50 (with pre-trained weights)

**How to Run the Application**

Follow these steps to get a local copy of the project up and running.

1. Clone the repository

git clone Nife77

cd skin-cancer-app

2. Set up the environment

Make sure you have Python 3.8 or higher installed. Then, install the required libraries.

pip install -r requirements.txt

3. Run the app

Once the dependencies are installed, you can start the Streamlit application.

streamlit run app.py

The app will open in your web browser, allowing you to upload an image and get a prediction.

**📅 Project Timeline**

This project was developed in a structured, phased approach to ensure all components were built and integrated efficiently.

Phase 1 (Weeks 1-2): Data Collection & Preparation

Phase 2 (Weeks 3-4): Model Development & Training

Phase 3 (Weeks 5-6): Application & XAI Integration

Phase 4 (Weeks 7-8): Finalization & Review

**🔗 Contact & Acknowledgments**

Developer: Udath Hegde

GitHub: github.com/nife77

This project was developed based on publicly available academic research and datasets, including those from the International Skin Imaging Collaboration (ISIC).
