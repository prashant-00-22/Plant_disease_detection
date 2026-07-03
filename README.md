# Plant Disease Detection (Flask + Transfer Learning)

Plant Disease Detection is a web application that predicts plant diseases from leaf images using a MobileNet-based transfer learning model.

> **Note (important):** The Flask app expects a trained model file at `models/plant_disease_model.h5`. If the model file is missing, the server will still start, but prediction requests will return an error.

---

## ✨ Features
- Upload a leaf image via a simple web UI
- Run inference using a TensorFlow/Keras model
- Display the predicted disease + per-class probabilities
- Robust routing + user-friendly error handling

---

## 📁 Project Structure
- `app.py` — Flask application (routes: `/`, `/about`, `/how-it-works`, `/upload`, `/result`)
- `templates/` — HTML templates for the UI
- `static/` — static assets + `static/uploads/` for user uploads
- `requirements.txt` — Python dependencies
- `models/plant_disease_model.h5` — trained model (**required at runtime**)
- `Code/` — notebooks and experiments used during development

---

## 🧰 Setup & Run Locally

### 1) Install dependencies
```bash
pip install -r Early-Plant-Disease-Detection-Using-Transfer-Learning/requirements.txt
```

### 2) (Required) Place the trained model
Copy your trained model to:
- `models/plant_disease_model.h5`

### 3) Start the server
Run:
```bash
python Early-Plant-Disease-Detection-Using-Transfer-Learning/app.py
```
Then open:
- `http://127.0.0.1:5000`

---

## ▶️ How to Use
1. Open the home page
2. Click **Upload** and select a leaf image (`.jpg`, `.jpeg`, `.png`, etc.)
3. Submit the form
4. View the predicted disease and confidence on the result page

---

## 🔧 Requirements
Check `Early-Plant-Disease-Detection-Using-Transfer-Learning/requirements.txt` for the full dependency list.

---

## 📌 License
Add your license information here (e.g., MIT, Apache-2.0) if you want it included on GitHub.

