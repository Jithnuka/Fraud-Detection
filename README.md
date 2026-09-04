# 🛡️ Fraud Risk Intelligence Console

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://fraud-detection-k9th4vbrnxark9okngx2dx.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

> A modern enterprise-grade financial fraud detection engine powered by a custom **PyTorch Graph Neural Network (GraphSAGE)** model and packaged inside a high-performance **Streamlit** executive console.

---

## 🌟 Overview

The **Fraud Risk Intelligence Console** delivers real-time transaction scoring, deep graph-based structural analysis, and batch risk auditing for financial institutions. Designed for compliance teams, fraud analysts, and risk executives, it bridges complex AI graph modeling with actionable operational decisions.

---

## ✨ Key Features

- ⚡ **Single Transaction Scoring**: Instant manual risk evaluation for high-value or flagged financial activities.
- 📦 **High-Volume CSV Batch Processing**: Chunked streaming pipeline designed to process large transaction files seamlessly.
- 🎯 **Adjustable Risk Thresholds**: Dynamic decision engine allowing analysts to tune precision/recall trade-offs on the fly.
- 📊 **Executive Analytics & ROC Curves**: Automatic generation of Probability Distributions, Confusion Matrices, and ROC-AUC curves for labeled evaluation data.
- 📥 **Exportable Audit Logs**: One-click CSV export tailored for review queues and audit workflows.

---

## 🏗️ Architecture & Project Structure

```text
Frud Detection/
├── 📁 app/                   # Streamlit app views and modular components
├── 📁 artifacts/             # Scalers, encoders, and pre-trained weights
├── 📁 data/                  # Dataset samples and raw inputs
├── 📁 models/                # PyTorch & GraphSAGE model architecture definitions
├── 📁 notebooks/             # Exploratory Data Analysis & training notebooks
├── 📁 src/                   # Core business logic, data preprocessing & graph pipeline
├── 📄 streamlit_app.py       # Main Streamlit Console entrypoint
└── 📄 requirements.txt       # Project dependencies
```

---

## 📊 Expected Data Schema

The model accepts transaction records containing the following financial feature vectors:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `step` | `int` | Unit of time in the simulation (1 step = 1 hour) |
| `type` | `string` | Transaction type (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`) |
| `amount` | `float` | Amount of the transaction in local currency |
| `oldbalanceOrg` | `float` | Initial balance of sender prior to transaction |
| `newbalanceOrig` | `float` | Updated balance of sender post transaction |
| `oldbalanceDest` | `float` | Initial balance of recipient prior to transaction |
| `newbalanceDest` | `float` | Updated balance of recipient post transaction |
| `isFraud` *(Optional)* | `int` | Ground truth target (0 = Legitimate, 1 = Fraudulent) |

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-username/fraud-detection-console.git
cd "Frud Detection"

# Create virtual environment
python -m venv .venv

# Activate environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Console

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser to access the dashboard.

---

## 🌐 Online Live Demo

Experience the deployed version instantly without local installation:
👉 **[Open Fraud Risk Intelligence Console](https://fraud-detection-k9th4vbrnxark9okngx2dx.streamlit.app/)**

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.


