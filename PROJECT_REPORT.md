# IDRBT Project Submission & Technical Dossier

**Domain Track:** AI, Data Analytics, Frontier AI Models and Emerging Scenarios  
**Core Theme:** Responsible, Explainable and Interpretable AI Models in the Banking Sector  
**Project Title:** **FairCredit-XAI: A Transparent, Fair, and Counterfactual Credit Risk Evaluation System for Indian Banking**  

---

## 1. Executive Summary

As Indian commercial banks and NBFCs accelerate digital lending under the aegis of the Digital Public Infrastructure (DPI)—incorporating Account Aggregators (AA), CIBIL/Experian bureaus, and UPI credit rails—machine learning models are increasingly deployed to automate underwriting. However, black-box AI architectures pose profound **systemic, regulatory, and constitutional challenges**:

1. **Regulatory Non-Compliance (Fair Lending & Adverse Action)**: Reserve Bank of India (RBI) mandates fair treatment, non-discrimination (under Articles 14 & 15 of the Constitution and Priority Sector Lending directives), and model explainability. Borrowers denied credit cannot simply be told *"Your application was rejected by the algorithm."*
2. **Hidden Demographic Bias & Disparate Impact**: Proxy variables in financial transaction data can introduce subtle, unintended bias against underserved demographics (e.g., rural vs. urban borrowers, self-employed individuals, women entrepreneurs).
3. **Absence of Actionable Recourse**: Traditional Explainable AI (XAI) tools (like generic feature importance) only provide backward-looking attribution without telling the customer **what specific, minimal steps they must take to achieve credit approval**.

**FairCredit-XAI** directly addresses this imperative with a lightweight, production-ready framework that provides:
- **Pillar 1: Automated Fair Lending & Bias Auditing**: Calculates Disparate Impact Ratios (DIR, 4/5ths Rule), Demographic Parity, and Equal Opportunity across demographic sub-groups.
- **Pillar 2: Transparent Decision Attribution**: Utilizes SHAP (SHapley Additive exPlanations) for both portfolio-wide global governance and applicant-specific local waterfall attributions.
- **Pillar 3: Actionable Counterfactual Recourse**: Generates an automated, RBI-compliant **Adverse Action Notice** that computes the minimal perturbation needed (e.g., pay down debt-to-income by 15%, extend tenure) to flip an adverse decision to an approved status.

---

## 2. Details of System Implementation & Architecture

```
                               ┌────────────────────────────────────────────────────────┐
                               │           Loan Origination System (LOS)                │
                               │  (Account Aggregator + Bureau + CBS Application Data)   │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │       Feature Engineering & Sensitive Exclusion        │
                               │   (Excludes Gender, Region, Castes from direct input)  │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │             Calibrated Credit Risk Engine              │
                               │     (Gradient Boosting / Random Forest Ensemble)       │
                               │              Predicts P(Approval / Default)            │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                        ┌─────────────────────────────────┼─────────────────────────────────┐
                        ▼                                 ▼                                 ▼
         ┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────────┐
         │     Module 1: Fairness &      │ │      Module 2: Local &        │ │  Module 3: Counterfactual     │
         │         Bias Auditor          │ │    Global XAI (SHAP Engine)   │ │      Recourse Generator       │
         ├───────────────────────────────┤ ├───────────────────────────────┤ ├───────────────────────────────┤
         │ • Disparate Impact Ratio      │ │ • Portfolio Feature Ranking   │ │ • Minimum Distance Solver     │
         │ • Demographic Parity Diff     │ │ • Underwriter SHAP Waterfall  │ │ • Actionable Financial Levers │
         │ • 4/5ths Rule Compliance Test │ │ • Positive / Negative Drags   │ │ • Adverse Action Notice Gen   │
         └───────────────────────────────┘ └───────────────────────────────┘ └───────────────────────────────┘
                        │                                 │                                 │
                        └─────────────────────────────────┼─────────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │         Unified Governance & Inspection Portal         │
                               │  (Live Streamlit / FastAPI / 1-Click Model Risk Card)  │
                               └────────────────────────────────────────────────────────┘
```

### Key Technical Specifications:
- **Core Engine:** Python 3.12, Scikit-Learn, SHAP, Pandas, NumPy.
- **Explainability Protocol:** TreeSHAP exact attribution algorithm computing localized log-odds contributions per applicant.
- **Fairness Metrics Standard:** Disparate Impact Ratio ($DIR \ge 0.80$), Demographic Parity Difference ($DPD \le 0.10$), Equal Opportunity Difference ($EOD \le 0.10$).
- **Deployment Interface:** Interactive Streamlit Dashboard with automated Markdown/PDF Model Card generation.

---

## 3. Technology Deployments & Open-Source Contributions

### A. Open-Source Contribution Blueprint
- **Repository Name:** `faircredit-xai-idrbt`
- **License:** MIT License (Permissive open-source research and banking deployment).
- **Architecture Modularity:** Clean, standalone modules (`credit_model.py`, `fairness_audit.py`, `explainability.py`, `app.py`) designed for zero-friction integration with existing Python/Java core banking services.
- **Reproducibility:** Self-contained data synthesis, deterministic seeds, and one-command launch via `streamlit run app.py`.

### B. Enterprise / Banking Deployment Strategy
1. **Containerized Microservice:**
   - Packaged as a Docker container running FastAPI with asynchronous endpoints:
     - `POST /api/v1/underwrite`: Evaluates applicant, returns approval probability, and invokes SHAP.
     - `POST /api/v1/recourse`: Triggers counterfactual solver for rejected applications.
     - `GET /api/v1/audit/fairness`: Audits batch production logs on a daily/weekly schedule.
2. **Integration with Core Banking Systems (CBS):**
   - Interfaced via JSON REST APIs or Apache Kafka event streams during loan processing workflows in Loan Origination Systems (LOS) like FinnOne, Finacle, or custom cloud banking platforms.
3. **Data Protection & Privacy:**
   - Operates entirely on pseudonymized customer identifiers (`applicant_id`). Sensitive demographic attributes are never used in score computation; they are retained solely in an isolated, encrypted audit enclave for regulatory bias audits.

---

## 4. Banking & Financial Sector Consulting Scenarios

This framework is directly applicable to three major real-world consulting scenarios in Indian banking:

### Scenario 1: Priority Sector Lending (PSL) & Financial Inclusion Audits
- **Challenge:** Indian public and private sector banks must meet stringent Priority Sector Lending (PSL) targets (e.g., agriculture, micro-enterprises, weaker sections). Algorithms trained on historical data often penalize borrowers lacking extensive credit bureau records (thin-file borrowers).
- **Consulting Solution:** Deploy FairCredit-XAI’s Disparate Impact module to detect indirect disparate impact against semi-urban and rural demographics. Rebalance decision boundaries dynamically to ensure PSL compliance without escalating Non-Performing Assets (NPAs).

### Scenario 2: RBI Model Risk Management (MRM) Compliance
- **Challenge:** Under RBI supervisory audits and Internal Rating-Based (IRB) approaches, banks must present comprehensive Model Validation and Model Risk Management reports.
- **Consulting Solution:** Automate the generation of **Standardized Model Cards** detailing out-of-sample AUC-ROC, feature stability (PSI), and SHAP attribution bounds, reducing annual model validation audit cycles from weeks to hours.

### Scenario 3: Customer Grievance Redressal & Adverse Action Notices
- **Challenge:** Rising borrower complaints filed with the RBI Banking Ombudsman regarding arbitrary digital loan rejections.
- **Consulting Solution:** Implement the **Counterfactual Recourse Engine** at the customer portal level. When a loan is denied, the customer receives clear, non-technical steps to improve their financial health (e.g., *"Reduce credit card utilization from 62% to 35% to become eligible"*), dramatically improving customer trust and financial literacy.

---

## 5. Verification & Documentary Evidence

| Deliverable | Location | Status |
| :--- | :--- | :--- |
| **Complete Source Code** | `d:/IDRBT banking project/` | Fully functional, documented, and modular |
| **Live Interactive UI** | `app.py` (Streamlit) | Real-time underwriter view, SHAP waterfall, and bias auditor |
| **Synthetic Banking Dataset** | `loan_data.csv` (2,500 records) | Mapped to Indian banking context (CIBIL, INR income, DTI) |
| **Regulatory Model Card** | Exportable from UI / `README.md` | Pre-configured to RBI Responsible AI standards |
| **Interview Talking Points** | `INTERVIEW_PITCH.md` | Structured 3-minute executive pitch and FAQ response bank |
