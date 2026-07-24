# Fraud Risk Intelligence Console

Live demo: https://fraud-detection-k9th4vbrnxark9okngx2dx.streamlit.app/

![Uploading Fruad Detection.png…]()


A premium Streamlit experience for financial fraud detection using a pre-trained PyTorch GraphSAGE model. The application combines real-time transaction scoring, batch analytics, and executive-friendly reporting in a polished enterprise interface.

## Highlights
- Modern dashboard-style UI with a premium enterprise aesthetic
- Single-transaction risk assessment for rapid review
- High-volume CSV batch scoring with chunked processing
- Interactive fraud probability analytics and downloadable reports
- Clear risk classification for operational decision-making

## What the app does
The system evaluates transaction features such as amount, balances, and payment type to estimate the chance that a transaction is fraudulent. It supports both manual inspection and bulk review workflows for analysts and risk teams.

## Key features
- Manual review mode for one-off transaction assessment
- Bulk analytics mode for uploading CSV files and processing them efficiently
- Probability distribution visualization
- Confusion matrix and ROC curve when labeled test data is supplied
- CSV export for predictions and review queues

## Tech stack
- Python
- Streamlit
- PyTorch
- scikit-learn
- pandas / numpy
- seaborn / matplotlib
- gdown

## Project structure
- streamlit_app.py: main application UI and prediction workflow
- model.py: model definition
- train.py: training workflow
- data_prep.py: preprocessing helpers
- evaluate.py: evaluation utilities
- requirements.txt: Python dependencies

## Installation
Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the app locally
```bash
streamlit run streamlit_app.py
```

## Input requirements
The app expects transaction data with the following fields:
- step
- type
- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest

For labeled evaluation, include an optional isFraud column. If present, the app can generate performance visualizations such as a confusion matrix and ROC curve.

## Usage guidance
1. Open the app in your browser.
2. Choose Manual Review for a single transaction or Bulk Analytics for a file upload.
3. Set your threshold for what should be considered suspicious.
4. Review the score, export results, and share reports with stakeholders.

## Notes
- The model artifact model.pt should be present in the project root.
- The preprocessing artifact is downloaded automatically on first run if it is not available locally.
