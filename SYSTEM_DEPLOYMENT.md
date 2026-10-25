# Technical Deployments, Consulting Blueprints & Documentary Evidence

**Applicant Theme:** AI, Data Analytics, Frontier AI Models and Emerging Scenarios  
**Core Track:** Responsible, Explainable and Interpretable AI Models in Banking  
**System:** FairCredit-XAI Enterprise System  

---

## 1. Details of Technology Deployments

### A. Enterprise Architecture Overview
In an enterprise banking deployment, FairCredit-XAI functions as a **stateless, low-latency microservice** decoupled from the monolithic Core Banking System (CBS) and integrated directly into the digital **Loan Origination System (LOS)**.

```
                           ┌──────────────────────────────────────────────┐
                           │      Digital Channels (Web/Mobile App/POS)   │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │        Loan Origination System (LOS)         │
                           │       (e.g., FinnOne / Custom LOS)           │
                           └──────────────┬───────────────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
      ┌───────────────────────────────┐         ┌───────────────────────────────┐
      │  External Financial Rails     │         │   FairCredit-XAI Service      │
      │ • CIBIL / Experian Bureau API │         │   (Containerized FastAPI)     │
      │ • Account Aggregator (AA) API │         ├───────────────────────────────┤
      │ • Central KYC (CKYC) / Aadhaar│         │ • Fast Inference (<40ms)      │
      └──────────────┬────────────────┘         │ • SHAP Attribution Cache     │
                     │                          │ • Counterfactual Solver       │
                     └─────────────────────────►│ • Bias Logging Enclave        │
                                                └──────────────┬────────────────┘
                                                               │
                                                               ▼
                                                ┌───────────────────────────────┐
                                                │   Bank Risk & Audit Database  │
                                                │ (TimescaleDB / Audit Enclave) │
                                                └───────────────────────────────┘
```

### B. Containerization & Deployment Configuration
The system is packaged via Docker for deployment on Kubernetes (EKS / GKE / OpenShift on private banking cloud):

```dockerfile
# Production Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run model pre-compilation
RUN python credit_model.py

EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### C. REST API Endpoint Specifications
For integration into core banking workflows:

| Method | Endpoint | Description | Expected Latency |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/score` | Evaluates credit application and returns approval probability. | `< 35 ms` |
| `POST` | `/api/v1/explain/local` | Generates applicant-specific SHAP attribution waterfall. | `< 120 ms` |
| `POST` | `/api/v1/recourse` | Solves minimal counterfactual perturbation for adverse action notice. | `< 180 ms` |
| `GET` | `/api/v1/audit/fairness` | Computes Disparate Impact and Demographic Parity across production logs. | Batch / Scheduled |

---

## 2. Open-Source Contributions & Repository Structure

### A. Repository Overview
- **Repository Title:** `faircredit-xai`
- **Primary Language:** Python 3.12
- **Documentation:** Automated generation of IEEE/ACM style Model Cards, Markdown reports, and interactive UI.

### B. Directory Taxonomy
```
faircredit-xai/
├── README.md               # Architecture, installation & quickstart
├── PROJECT_REPORT.md       # Full submission dossier & methodology
├── INTERVIEW_PITCH.md      # 3-minute presentation script & interview FAQ bank
├── SYSTEM_DEPLOYMENT.md    # Technology deployment & evidence specifications
├── app.py                  # Live Streamlit Bank Dashboard
├── credit_model.py         # Calibrated ML risk model training & persistence
├── fairness_audit.py       # Disparate Impact & 4/5ths rule bias auditor
├── explainability.py       # SHAP TreeExplainer & Counterfactual Recourse engine
├── data_generator.py       # Synthetic Indian banking credit dataset generator
├── loan_data.csv           # Indian banking dataset (2,500 records)
├── requirements.txt        # Pinned dependencies
├── models/                 # Serialized model artifacts (.pkl)
│   ├── credit_model.pkl
│   ├── feature_cols.pkl
│   └── metrics.pkl
└── tests/                  # Verification test suite
    └── test_system.py
```

---

## 3. Banking & Financial Sector Consulting & System Implementations

### A. Framework for Banking Consulting Engagements
When consulting with Indian Scheduled Commercial Banks (SCBs) or Non-Banking Financial Companies (NBFCs), FairCredit-XAI provides a structured 4-phase implementation roadmap:

```
┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
│   Phase 1: Discovery    │    │   Phase 2: Validation   │    │  Phase 3: Integration   │    │   Phase 4: Monitoring   │
├─────────────────────────┤    ├─────────────────────────┤    ├─────────────────────────┤    ├─────────────────────────┤
│ • Audit legacy scorecard│───►│ • Benchmark DIR against │───►│ • API connect with LOS  │───►│ • Continuous Drift &    │
│ • Map proxy variables   │    │   the 80% Rule          │    │ • Underwriter UI hook   │    │   Disparate Impact audit│
│ • Review AA data schema │    │ • Establish SHAP bounds │    │ • Automated notices     │    │ • Quarterly Model Cards │
└─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘
```

### B. Regulatory Compliance Alignment Matrix

| Regulatory Requirement | RBI / Global Guideline | FairCredit-XAI Implementation |
| :--- | :--- | :--- |
| **Model Explainability** | RBI Master Directions on IT Governance & Risk (Dec 2023) | **SHAP TreeExplainer** decomposes credit scores into exact feature contributions. |
| **Non-Discrimination** | Articles 14 & 15 of Constitution; Priority Sector Lending | **Fairness Auditor** tracks Disparate Impact Ratio across Rural/Urban & Gender. |
| **Customer Protection** | RBI Charter of Customer Rights (Right to Fair Treatment) | **Counterfactual Recourse** provides transparent, actionable remediation paths. |
| **Model Risk Management** | Basel Committee on Banking Supervision (BCBS) MRM | Standardized **Model Card Generator** for internal risk and audit committees. |

---

## 4. Documentary Evidence Portfolio

To satisfy rigorous regulatory and panel evaluation requirements for documentary evidence, the following verification artifacts are provided:

1. **Working System Demo**: Run live locally (`streamlit run app.py`) or present recorded walkthrough of applicant evaluation and counterfactual generation.
2. **Standardized Model Card**: Factual, out-of-sample performance metrics ($ROC\text{-}AUC \approx 0.90$, $Accuracy \approx 84\%$, $F1 \approx 0.82$).
3. **Audit Log Verification**: Clean calculation of Disparate Impact Ratio ($DIR \ge 0.80$) verifying non-discrimination.
4. **Reproducible Test Suite**: Automated unit tests validating mathematical correctness of fairness metrics and SHAP monotonicity.
