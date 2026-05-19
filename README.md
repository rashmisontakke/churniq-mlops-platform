# ChurnIQ — Customer Retention Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-3.12-blue)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?logo=streamlit)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An end-to-end AI platform that predicts customer churn, explains why it happens, and recommends personalised retention actions — built with production-grade MLOps practices.

---

## 🎯 Business Problem

Customer churn costs banks and financial institutions millions annually. Acquiring a new customer costs **5–10× more** than retaining an existing one. Banks need to know:

- **Who** is about to leave — before they leave
- **Why** they are leaving — explainable, not a black box
- **What to do** — specific, prioritised retention actions
- **How much revenue is at risk** — business impact in ₹

ChurnIQ answers all four questions in a single API call.

---

## 🏦 Indian Banking Relevance

| Dataset Feature | Indian Banking Equivalent |
|---|---|
| CreditScore (300–900) | CIBIL Score — used by every Indian bank |
| IsActiveMember | RBI active account definition |
| Balance = 0 | NPA risk indicator per RBI IRACP norms |
| NumOfProducts | Cross-sell penetration rate |
| Geography | Indian state / zone mapping |

**SHAP explainability** directly supports RBI's model risk management guidelines, which require banks to explain credit and retention decisions to customers. This is a compliance requirement, not optional.

**Target industry:** HDFC Bank · Axis Bank · Bajaj Finance · Paytm · PhonePe · TCS Banking BU · Infosys Finacle · Mu Sigma · Fractal Analytics

---

## 🏗️ Architecture

```
Data (CSV)
    │
    ▼
feature_engineering.py ──── 6 custom features including NPA risk flag
    │
    ▼
sklearn ImbPipeline
    ├── StandardScaler (numerical)
    ├── OneHotEncoder (categorical)
    ├── SMOTE (class imbalance — training only)
    └── XGBoost Classifier
    │
    ▼
MLflow ─── experiment tracking · model registry · artifact storage
    │
    ├──► pipeline.pkl  ──► FastAPI (/predict · /predict/batch · /health · /metrics)
    ├──► explainer.pkl ──► SHAP explanations per prediction
    └──► metrics.json  ──► Streamlit dashboard
    │
    ▼
Streamlit Dashboard
    ├── Tab 1: Overview      — churn distribution, geography, age analysis
    ├── Tab 2: Risk Scorer   — live prediction + gauge + retention actions
    ├── Tab 3: SHAP Analysis — feature impact per customer
    └── Tab 4: Cohort Intel  — heatmaps, revenue at risk by segment
```

---

## ✨ What Makes This Different

**Most churn projects:** train model → print accuracy → done.

**ChurnIQ:**

- Returns **churn probability + SHAP top features + retention actions** in one API response
- Uses **sklearn ImbPipeline** — SMOTE only on training data, no leakage
- **Temporal split** (80/20 ordered) — no random leakage of future data into training
- **6 original engineered features** — `balance_salary_ratio`, `npa_risk_flag`, `engagement_score` etc.
- **Rule-based retention engine** — maps SHAP feature contributions to business actions
- **Business impact calculation** — revenue at risk and estimated savings per prediction
- **MLflow Model Registry** — proper staging/production model governance
- **Evidently AI** — data drift monitoring with HTML reports

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 86.0% |
| ROC-AUC | 0.8712 |
| F1 Score | 0.6199 |
| Recall | Tuned for churn detection |

> Model: XGBoost · Dataset: 10,000 customers · Churn rate: 20.4% · Split: temporal 80/20

---

## 🔧 Feature Engineering

Original features added beyond the raw dataset:

| Feature | Formula | Business Meaning |
|---|---|---|
| `balance_salary_ratio` | Balance / (Salary + 1) | Financial health indicator |
| `products_per_tenure` | Products / (Tenure + 1) | Engagement rate over time |
| `age_bucket` | pd.cut into 4 groups | Age-based risk segment |
| `is_zero_balance` | Balance == 0 | Strongest single churn signal |
| `engagement_score` | Active × Products × HasCard | Composite loyalty score |
| `npa_risk_flag` | Zero balance + inactive + 1 product | Mirrors RBI stressed account classification |

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| ML Model | XGBoost 3.2 | Industry standard for tabular data |
| Explainability | SHAP 0.51 | RBI compliance + per-prediction explanations |
| Pipeline | sklearn ImbPipeline | Prevents SMOTE data leakage |
| Class Imbalance | SMOTE | Handles 80/20 class split |
| Experiment Tracking | MLflow 3.12 | Model registry + artifact storage |
| Orchestration | Prefect 3.7 | Training + monitoring flows |
| Drift Detection | Evidently AI | HTML drift reports |
| API | FastAPI 0.136 + Pydantic v2 | Async, typed, auto-documented |
| Dashboard | Streamlit 1.57 + Plotly | Live demo, 4-tab UI |
| Deployment | Docker + AWS Lambda | Serverless inference |
| CI/CD | GitHub Actions | Automated test + deploy |

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/rashmisontakke/ChurnIQ.git
cd ChurnIQ

# Setup
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

# Train model (creates pipeline.pkl + explainer.pkl)
python src/train.py

# Start API — Terminal 1
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload

# Start dashboard — Terminal 2
streamlit run src/streamlit_app.py
```

Open:
- Dashboard → `http://localhost:8501`
- API docs  → `http://localhost:8000/docs`

---

## 📡 API Reference

### POST `/predict`

```json
// Request
{
  "geography": "Germany",
  "gender": "Female",
  "age": 42,
  "credit_score": 600,
  "tenure": 3,
  "balance": 0.0,
  "estimated_salary": 100000.0,
  "num_of_products": 1,
  "has_cr_card": 1,
  "is_active_member": 0,
  "customer_id": "CUST_001"
}

// Response
{
  "customer_id": "CUST_001",
  "churn_probability": 0.8431,
  "risk_label": "HIGH",
  "confidence": 0.8431,
  "top_risk_factors": [
    {
      "feature": "is_zero_balance",
      "shap_value": 0.4821,
      "direction": "increases_risk",
      "impact_pct": 34.2
    }
  ],
  "retention_actions": [
    {
      "priority": 1,
      "action": "Offer savings account with 4.5% APY",
      "reason": "Zero balance is the strongest churn predictor",
      "impact": "High"
    }
  ],
  "business_impact": {
    "revenue_at_risk_inr": 42155.0,
    "estimated_saveable_inr": 12646.5
  },
  "model_version": "2.0.0",
  "prediction_id": "uuid-here"
}
```

Other endpoints: `GET /health` · `GET /metrics` · `POST /predict/batch`

Full interactive docs: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
ChurnIQ/
├── src/
│   ├── train.py                # XGBoost pipeline + SHAP + MLflow
│   ├── serve.py                # FastAPI — /predict, /batch, /health
│   ├── streamlit_app.py        # 4-tab Streamlit dashboard
│   ├── feature_engineering.py  # 6 custom engineered features
│   ├── monitor.py              # Evidently drift detection
│   ├── pipeline.py             # Prefect orchestration flows
│   └── lambda_handler.py       # AWS Lambda handler
├── data/
│   └── Customer-Churn-Records.csv
├── models/                     # Generated after training
├── tests/
│   ├── test_train.py
│   └── test_integration.py
├── .github/workflows/
│   ├── ci.yml
│   └── deploy-lambda.yml
├── Dockerfile
├── Dockerfile.fastapi
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🔄 Makefile Commands

```bash
make train      # Train XGBoost + save SHAP explainer + log to MLflow
make batch      # Batch predictions on full dataset
make monitor    # Evidently drift report → reports/drift_report.html
make test       # Run pytest suite
make api        # Start FastAPI server
make dashboard  # Start Streamlit dashboard
```

---

## 📈 Monitoring

Evidently AI generates HTML drift reports comparing training vs current data.

```bash
python src/monitor.py
# Output: reports/drift_report.html
```

Monitors: data drift across 10 features · model performance stability · churn rate trends

---

## 🚢 Deployment

### Docker
```bash
docker build -f Dockerfile.fastapi -t churniq-api .
docker run -p 8000:8000 churniq-api
```

### AWS Lambda
Push to `main` → GitHub Actions builds image → pushes to ECR → deploys Lambda automatically.

---

## 👩‍💻 Author

**Rashmi Sontakke**

[![GitHub](https://img.shields.io/badge/GitHub-rashmisontakke-black?logo=github)](https://github.com/rashmisontakke)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
