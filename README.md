# FairCredit-XAI: Responsible, Explainable and Interpretable AI for Credit Decisioning in Indian Banking
### Aligned with RBI Responsible AI Guidelines & Model Risk Management (MRM)

---

## 📌 Executive Summary
**FairCredit-XAI** is a lightweight, production-grade, and regulatory-compliant framework designed for Indian commercial banks and NBFCs. It bridges the gap between high-performing machine learning models (Gradient Boosted Trees) and stringent regulatory requirements for **Fair Lending, Adverse Action Explanation, and Bias Auditing**.

```
                           ┌──────────────────────────────────────────────┐
                           │      Loan Application / CBS Data             │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │     Credit Risk Model        │
                                   │  (LightGBM / Balanced GBDT)  │
                                   └──────────────┬───────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                             ▼                             ▼
       ┌────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
       │   Pillar 1: Fairness   │   │   Pillar 2: Explainable  │   │  Pillar 3: Actionable    │
       │     & Bias Auditing    │   │        AI (SHAP)         │   │   Counterfactual Recourse│
       ├────────────────────────┤   ├──────────────────────────┤   ├──────────────────────────┤
       │ • Disparate Impact     │   │ • Global Risk Drivers    │   │ • "What needs to change  │
       │ • Demographic Parity   │   │ • Local Applicant SHAP   │   │    for approval?"        │
       │ • RBI Fair Lending     │   │ • Feature Attributions   │   │ • Compliant Adverse      │
       │   Compliance Score     │   │   for Underwriters       │   │   Action Notice          │
       └────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
                    │                             │                             │
                    └─────────────────────────────┼─────────────────────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │   1-Click Regulatory Audit   │
                                   │  Model Card & Bank Dashboard │
                                   └──────────────────────────────┘
```

---

## 🎯 Strategic Industry Relevance

1. **Strategic Fit for Indian Banking & RBI Compliance**:
   - Regulatory focus is squarely on **Responsible AI**, **Fair Credit Access**, and **Customer Protection**, meaning black-box loan denials without actionable reasons are unacceptable.
   - This project addresses the exact real-world dilemma: *How do Indian banks deploy predictive AI without violating non-discrimination principles or leaving customers uninformed?*

2. **The 3-Minute Interview Pitch**:
   - **The Problem**: *"Traditional AI models reject loans with an opaque score (e.g. 0.42). Regulators require banks to prove the model isn't biased against protected demographics (e.g., gender, rural regions), and customers have a right to know specifically why they were denied and how to become eligible."*
   - **The Innovation**: *"FairCredit-XAI introduces a unified 3-tier auditing pipeline: (1) Automated Bias Auditing using Disparate Impact & Equalized Odds, (2) SHAP-based Local Explainability for underwriters, and (3) Actionable Counterfactual Recourse that generates a personalized RBI-compliant Adverse Action Notice telling the applicant the exact minimal steps needed to qualify."*
   - **The Practical Impact**: *"Reduces model governance cycle from weeks to minutes, provides verifiable compliance evidence for RBI inspection, and promotes financial inclusion."*

---

## 🏛️ System Architecture & Workflow

### 1. Data Ingestion & Preprocessing
- Realistic Indian banking variables:
  - `cibil_score`: Credit bureau score (300 - 900)
  - `monthly_income_inr`: Monthly disposable income in ₹
  - `debt_to_income_ratio`: Monthly debt obligations / gross income
  - `loan_amount_inr`: Requested loan amount
  - `tenure_months`: Loan repayment tenure
  - `existing_loans_count`: Number of active credit lines
  - `demographic_region`: Urban, Semi-Urban, Rural (Sensitive attribute for bias audit)
  - `gender`: Male, Female (Sensitive attribute for fair lending)
  - `employment_type`: Salaried, Self-Employed, Gig Worker

### 2. Credit Risk Engine
- High-performance, calibrated classifier (RandomForest/GradientBoosting) predicting loan default probability and creditworthiness.

### 3. Pillar 1: Responsible AI & Fairness Audit
- **Disparate Impact Ratio (DIR)**: Checks whether the approval rate for unprivileged groups is at least 80% (4/5ths rule) of privileged groups:
  $$\text{DIR} = \frac{P(\hat{Y}=1 | D=\text{Rural})}{P(\hat{Y}=1 | D=\text{Urban})}$$
- **Demographic Parity Difference**: Difference in selection rates between groups.
- **Equal Opportunity**: Compares True Positive Rates across demographic segments.

### 4. Pillar 2: Explainable AI (XAI)
- **Global Explainability**: Summary plots showing top macro-factors influencing the bank's portfolio (e.g., Debt-to-Income, CIBIL, Income).
- **Local Explainability (SHAP Waterfall)**: Exact breakdown for a specific applicant. Explains to the bank loan officer which factors pushed the score towards rejection vs approval.

### 5. Pillar 3: Actionable Counterfactual Recourse (Adverse Action Notice)
- Instead of just stating *"Applicant denied"*, the recourse engine computes the nearest achievable state:
  > *"Loan Status: REJECTED (Probability 41%, Threshold 60%).*  
  > *Actionable Path to Approval:*  
  > *1. Reduce Debt-to-Income ratio from 54% to 38% (e.g., pay down ₹25,000 credit card balance)*  
  > *2. Increase loan tenure from 24 to 36 months to lower monthly EMI burden.*  
  > *Projected Approval Probability upon changes: 67% (APPROVED)."*

---

## 🚀 Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/faircredit-xai.git
cd faircredit-xai

# Install dependencies
pip install -r requirements.txt

# Run the interactive Streamlit Bank Dashboard
streamlit run app.py
```

---

## 📂 Project Structure
```
├── app.py                      # Interactive Bank Dashboard (Fairness + XAI + Recourse)
├── credit_model.py             # ML Model Training, Evaluation, and Serialization
├── fairness_audit.py           # Bias Auditing & Disparate Impact Metrics
├── explainability.py           # SHAP & Counterfactual Explanations Engine
├── data_generator.py           # Synthetic Indian Banking Loan Dataset Generator
├── requirements.txt            # Dependency list
├── PROJECT_REPORT.md           # Formal Project Report for Interview Submission
├── INTERVIEW_PITCH.md          # 3-Minute Interview Talking Script & FAQ Answers
├── SYSTEM_DEPLOYMENT.md        # Technology Deployment, Architecture & Evidence Guide
└── assets/                     # Visual diagrams and reports
```
