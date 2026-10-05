# 🚀 VitaMap — AI Resume Classification

> **An intelligent resume classification system that maps raw resume text to the most appropriate career category using NLP and Machine Learning.**

VitaMap is an end-to-end AI-powered resume classification system built for the **ResumeForge AI Hackathon**. The system takes an unseen resume as input, processes the text using NLP techniques, extracts meaningful features, and predicts the most appropriate resume category using trained machine learning models.

---

## 📌 Table of Contents
- [What is VitaMap?](#-what-is-vitamap)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🔌 API Reference](#-api-reference)
- [🧠 ML Pipeline](#-machine-learning-pipeline)
- [🧪 Evaluation](#-evaluation)
- [💻 Getting Started](#-running-the-project)
- [👥 Team & Contributors](#-team)
- [🔮 Future Roadmap](#-future-improvements)

---

## 🚀 What is VitaMap?

Recruiters and career platforms often receive large volumes of resumes spanning different roles and domains. VitaMap automates the first stage of this pipeline:

```text
Resume ➔ Text Preprocessing ➔ Feature Extraction ➔ ML Model Inference ➔ Predicted Category
```

The application provides a simple web interface where users can paste or upload resume text to instantly receive:
* **Predicted category** alongside classification confidence scores (when supported).
* **Key terms** contributing heavily to the model's final prediction.
* **Live metrics** including model evaluation history and confusion matrices.

---

## ✨ Features

* **🤖 AI Resume Classification:** Classifies unseen text into targeted occupational categories using ML.
* **🧹 NLP Preprocessing:** A custom pipeline that sanitizes text while preserving crucial technical tokens (e.g., `Python`, `C++`, `SQL`, `AWS`, `TensorFlow`, `Java`, `.NET`, `NLP`).
* **📊 Model Evaluation:** Built-in benchmarking evaluating Accuracy, Precision, Recall, F1-Scores (Macro/Weighted), and Confusion Matrices.
* **🔬 Model Comparison:** Evaluates Classical ML frameworks (`TF-IDF → Linear SVM`) against Neural NLP models (`Word2Vec → LSTM / GRU / Dense`).
* **🖥️ Guided Frontend Pipeline:** Visually tracks the processing state from ingestion down to final model inference.
* **📈 Real Intelligence Dashboard:** Displays authentic verification metrics drawn directly from model outputs rather than placeholder values.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      User Resume    │
                    │   Paste / Upload    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   React Frontend    │
                    │      VitaMap UI     │
                    └──────────┬──────────┘
                               │
                         HTTP / REST API
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI         │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Text Preprocessing  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Extraction  │
                    │     TF-IDF /        │
                    │     Word2Vec        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Trained ML Models   │
                    │ SVM / Neural Model  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction + Metrics│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     VitaMap UI      │
                    │ Category + Score    │
                    └─────────────────────┘
```

---

## 🛠️ Tech Stack

### Frontend
* React 19 / TypeScript / Vite
* Tailwind CSS v4 / Lucide React

### Backend & Machine Learning
* Python / FastAPI
* Scikit-learn / Word2Vec
* Keras & TensorFlow (LSTM / GRU / Dense Layers)

---

## 📁 Project Structure

```bash
VitaMap---ResumeForge-Hackathon/
├── backend/
│   ├── api/
│   │   └── main.py          # FastAPI application entrypoint
│   ├── data/                # Datasets and raw files
│   ├── models/              # Serialized trained model binaries
│   ├── notebooks/           # Jupyter notebooks for EDA & training
│   ├── src/                 # Core processing source modules
│   ├── tests/               # Test suites
│   └── requirements.txt     # Python backend dependencies
├── vitamap-frontend/
│   ├── src/
│   │   ├── components/      # UI Elements & Dashboards
│   │   ├── lib/             # Utilities and API clients
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json         # Frontend configuration & dependencies
│   └── vite.config.ts       # Vite build configurations
└── README.md                # Root project documentation
```

---

## 🔌 API Reference

### 1. Predict Resume Category
* **Endpoint:** `POST /predict`
* **Payload Format:** `application/json`

**Request Example:**
```json
{
  "resume_text": "Experienced Python developer with expertise in Django, SQL, AWS and machine learning..."
}
```

**Response Example:**
```json
{
  "category": "Data Science",
  "confidence": "0.92",
  "model": "TF-IDF + Linear SVM",
  "top_terms": [
    "python",
    "machine learning",
    "sql",
    "tensorflow"
  ]
}
```
> ⚠️ *Note: `confidence` metrics are only returned when natively supported by the underlying active backend model.*

### 2. Fetch Model Metrics
* **Endpoint:** `GET /metrics`
* **Response Example:**
```json
{
  "accuracy": 0.91,
  "macro_f1": 0.89,
  "weighted_f1": 0.90,
  "models": []
}
```

---

## 🧠 Machine Learning Pipeline

VitaMap implements a rigid 10-step modular engineering pipeline:
1. **Data Collection** ➔ 2. **EDA** ➔ 3. **Text Preprocessing** ➔ 4. **Feature Engineering** ➔ 5. **Train/Val/Test Split** ➔ 6. **Model Training** ➔ 7. **Evaluation** ➔ 8. **Error Analysis** ➔ 9. **Model Serialization** ➔ 10. **Production Inference**

* **Exploratory Data Analysis:** Full statistical audits are performed covering structural text length distribution, N-grams, class imbalances, missing entries, and class-specific vocabulary densities.
* **Data Leakage Mitigation:** Splitting protocols are explicitly completed *prior* to fitting vectorizers or scalers, ensuring validation steps remain completely isolated from training parameters.

---

## 🧪 Evaluation

System variants are aggressively tested across structural indicators rather than baseline accuracy alone:

| Metric | Target Evaluation Objective |
| :--- | :--- |
| **Accuracy** | Overall raw classification correctness |
| **Precision** | Correctness ratio of positive class predictions |
| **Recall** | True positive capture capability across categories |
| **F1 Score** | Harmonic mean balance between precision and recall flags |
| **Macro-F1** | Computes unweighted class metrics (equal class significance) |
| **Weighted-F1** | Computes metrics adjusted by sample support distribution size |
| **Confusion Matrix** | Deep class-level misclassification error tracking |

---

## 💻 Running the Project

### 1. Clone the Repository
```bash
git clone https://github.com
cd VitaMap---ResumeForge-Hackathon
```

### 2. Backend Environment Setup
```bash
cd backend
```
Create and launch your Python virtual environment:
* **Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

Install dependencies and start the Uvicorn engine:
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```
* Backend local endpoint running at: `http://localhost:8000`

### 3. Frontend Environment Setup
Open a separate terminal window and navigate to the UI module:
```bash
cd vitamap-frontend
npm install
```

Configure local configurations by creating a `.env` file in the frontend root:
```env
VITE_API_URL=http://localhost:8000
VITE_USE_MOCK_API=false
```

Boot up the local Vite engine:
```bash
npm run dev
```
* Frontend application local host running at: `http://localhost:5173`

---

## 🧑‍💻 Development Mode & Simulation

For rapid UI/UX adjustments independent of the ML service layer, you can run the UI using local mock data:
env VITE_USE_MOCK_API=true 
Ensure this flag is reverted to false for complete integration testing to communicate directly with live models.

## 👥 Team

Proudly engineered for the **ResumeForge AI Hackathon**:
* **Khushi** — Backend / Machine Learning / NLP Engineering
* **Have Patel** — Frontend / UI Architecture & System Integration

---

## 🔮 Future Improvements

Planned roadmap tracks outside of the core hackathon specifications:
* **Multi-Format Parsing:** Native extraction support for `.pdf` and `.docx` files.
* **Advanced Architectures:** Multi-variant Transformer additions (e.g., `BERT`, `RoBERTa` embeddings).
* **Explainable AI (XAI):** Visual overlays using integrated `LIME` / `SHAP` frameworks.
* **Enterprise Features:** Scalable batch resume ingestions and comprehensive recruiter analytics tools.

---

## 📜 License

Developed exclusively as an open-source prototype entry for the **ResumeForge AI Hackathon**.
