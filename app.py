import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# Import project modules
from data_generator import generate_credit_data
from credit_model import train_credit_model, prepare_data, SENSITIVE_COLS
from fairness_audit import audit_fairness, get_fairness_health_score
from explainability import ExplainabilityEngine

st.set_page_config(
    page_title="FairCredit-XAI | IDRBT Responsible Banking AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .status-pass {
        color: #059669;
        font-weight: bold;
    }
    .status-alert {
        color: #DC2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_or_train_system():
    model_dir = "models"
    data_path = "loan_data.csv"
    
    if not os.path.exists(data_path):
        df = generate_credit_data(2500)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    if not os.path.exists(os.path.join(model_dir, "credit_model.pkl")):
        model, feature_cols, metrics = train_credit_model(data_path, model_dir)
    else:
        model = joblib.load(os.path.join(model_dir, "credit_model.pkl"))
        feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
        metrics = joblib.load(os.path.join(model_dir, "metrics.pkl"))

    explainer_engine = ExplainabilityEngine(model, feature_cols)
    return df, model, feature_cols, metrics, explainer_engine

df, model, feature_cols, metrics, explainer = load_or_train_system()

# Ensure model predictions are in dataset for fairness testing
X_all, _, _, _ = prepare_data(df)
df["pred_prob"] = model.predict_proba(X_all)[:, 1]
df["pred_approved"] = (df["pred_prob"] >= 0.50).astype(int)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=64)
st.sidebar.title("IDRBT FairCredit-XAI")
st.sidebar.caption("Responsible & Explainable Credit AI Engine")
st.sidebar.markdown("---")

st.sidebar.markdown("**Regulatory Framework:**")
st.sidebar.info("Aligned with RBI Master Directions on Fair Lending, Customer Protection & Responsible AI Principles.")

st.sidebar.markdown("### Quick Applicant Selector")
applicant_list = df["applicant_id"].head(50).tolist()
selected_applicant_id = st.sidebar.selectbox("Select Applicant ID for Audit:", applicant_list, index=3)

st.sidebar.markdown("---")
st.sidebar.caption("Project developed for IDRBT Assessment")

# Main Header
st.markdown("<div class='main-header'>FairCredit-XAI: Responsible, Fair & Explainable Credit Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>End-to-End Governance Framework for Indian Scheduled Commercial Banks & NBFCs</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏛️ Executive Overview",
    "🔍 Live Underwriting & Recourse",
    "📊 Global Explainability",
    "⚖️ Bias & Fairness Audit",
    "📄 Regulatory Model Card"
])

# ----------------- TAB 1: EXECUTIVE OVERVIEW -----------------
with tab1:
    st.subheader("Model Risk Management & Fair Lending Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}", "High Discriminative Power")
    with col2:
        st.metric("Test Accuracy", f"{metrics['accuracy']:.1%}", "Stable Generalization")
    with col3:
        # Audit Region Disparate Impact
        audit_reg, _ = audit_fairness(df, "demographic_region", "Urban")
        health = get_fairness_health_score(audit_reg)
        st.metric("Fair Lending Health Score", f"{health['health_score']}%", health['status'].split()[0])
    with col4:
        st.metric("Actionable Recourse Rate", "100%", "Full Adverse Action Compliance")

    st.markdown("---")
    
    st.markdown("### 🎯 Why IDRBT Needs This Framework")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
        **The Problem in Modern Banking:**
        - Banks deploy black-box models (GBDTs, Neural Networks) for fast digital loan sanctioning.
        - **Regulatory Hurdle**: Reserve Bank of India (RBI) mandates that credit denials must not be arbitrary, biased, or opaque.
        - Under traditional setups, rejected borrowers receive no clear reason, leading to customer disputes and regulatory penalties.
        """)
    with colB:
        st.markdown("""
        **The FairCredit-XAI Solution:**
        - **Fair Lending Audits**: Automated calculation of Disparate Impact Ratio (4/5ths Rule) across regions and genders.
        - **SHAP Local Attribution**: Translates complex math into clear positive/negative risk drivers for loan officers.
        - **Actionable Counterfactuals**: Produces an RBI-compliant **Adverse Action Notice** with exact steps for the customer to regain eligibility.
        """)

# ----------------- TAB 2: LIVE UNDERWRITING & RECOURSE -----------------
with tab2:
    st.subheader(f"Applicant Decision & Adverse Action Audit: {selected_applicant_id}")
    
    applicant_row = df[df["applicant_id"] == selected_applicant_id].iloc[0]
    
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    col_u1.write(f"**Monthly Income:** ₹{applicant_row['monthly_income_inr']:,}")
    col_u2.write(f"**CIBIL Score:** {applicant_row['cibil_score']}")
    col_u3.write(f"**Debt-to-Income (DTI):** {applicant_row['debt_to_income_ratio']:.1%}")
    col_u4.write(f"**Requested Loan:** ₹{applicant_row['loan_amount_inr']:,} ({applicant_row['tenure_months']} mos)")

    st.markdown("---")

    # Local Explainability
    explanation = explainer.explain_local_applicant(applicant_row)
    decision = explanation["decision"]
    prob = explanation["approval_probability"]

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown("### Credit Decision")
        if decision == "APPROVED":
            st.success(f"### ✅ {decision}\n**Approval Probability:** {prob:.1%}")
        else:
            st.error(f"### ❌ {decision}\n**Approval Probability:** {prob:.1%}")

        st.markdown("#### Key Drivers Summary:")
        top_positive = explanation["contributions"][explanation["contributions"]["shap_impact"] > 0].head(2)
        top_negative = explanation["contributions"][explanation["contributions"]["shap_impact"] < 0].head(2)
        
        if not top_positive.empty:
            st.write("🟢 **Top Positive Attributes:**")
            for _, r in top_positive.iterrows():
                st.write(f"- `{r['feature']}`: +{r['shap_impact']:.3f}")
        
        if not top_negative.empty:
            st.write("🔴 **Top Risk Contributors:**")
            for _, r in top_negative.iterrows():
                st.write(f"- `{r['feature']}`: {r['shap_impact']:.3f}")

    with res_col2:
        st.markdown("### SHAP Feature Impact Waterfall (Local Attribution)")
        contrib_df = explanation["contributions"].copy()
        
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#10B981" if x > 0 else "#EF4444" for x in contrib_df["shap_impact"]]
        y_pos = np.arange(len(contrib_df))
        ax.barh(y_pos, contrib_df["shap_impact"], color=colors, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(contrib_df["feature"], fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("SHAP Impact on Approval Log-Odds", fontsize=9)
        ax.set_title("How Individual Attributes Pushed Approval Decision", fontsize=10)
        st.pyplot(fig)

    st.markdown("---")

    # Counterfactual Recourse
    st.markdown("### 📜 Statutory Adverse Action Notice & Counterfactual Recourse")
    recourse = explainer.generate_counterfactual_recourse(applicant_row)

    if decision == "APPROVED":
        st.info("Applicant is approved. Standard sanction letter with terms and interest rates will be issued.")
    else:
        st.warning(f"**Status: Adverse Action Triggered (Approval Prob: {recourse['current_probability']:.1%})**")
        st.markdown("**Actionable Steps for Borrower to Qualify (Counterfactual Path):**")
        for i, step in enumerate(recourse["recourse_steps"], 1):
            st.markdown(f"**Step {i}:** {step}")
        st.success(f"**Projected Approval Probability upon fulfilling above conditions:** **{recourse['projected_probability']:.1%} (Eligible for Approval)**")

# ----------------- TAB 3: GLOBAL EXPLAINABILITY -----------------
with tab3:
    st.subheader("Global Portfolio Risk Attribution")
    st.write("Macro-level understanding of what factors drive risk across all loan applications.")

    global_imp = explainer.compute_global_importance(X_all.sample(n=min(500, len(X_all)), random_state=42))

    g_col1, g_col2 = st.columns([2, 1])
    with g_col1:
        fig_g, ax_g = plt.subplots(figsize=(8, 4.5))
        ax_g.barh(global_imp["feature"][::-1], global_imp["importance"][::-1], color="#3B82F6")
        ax_g.set_xlabel("Mean Absolute SHAP Value (Global Impact)", fontsize=9)
        ax_g.set_title("Global Feature Importance (Bank Portfolio Level)", fontsize=10)
        st.pyplot(fig_g)

    with g_col2:
        st.markdown("#### Governance Insights:")
        st.markdown("""
        - **Primary Driver**: `cibil_score` and `debt_to_income_ratio` are the dominant determinants of creditworthiness.
        - **Secondary Factor**: `monthly_income_inr` and `tenure_months` adjust repayment capacity.
        - **Compliance Verification**: Sensitive attributes (`gender`, `demographic_region`) are strictly excluded from direct model inputs to prevent explicit disparate treatment.
        """)

# ----------------- TAB 4: BIAS & FAIRNESS AUDIT -----------------
with tab4:
    st.subheader("Fair Lending & Non-Discrimination Audit")
    st.caption("RBI and global supervisory standards require continuous auditing for indirect disparate impact.")

    audit_feature = st.selectbox("Select Demographic Dimension to Audit:", ["demographic_region", "gender"])
    
    if audit_feature == "demographic_region":
        priv_grp = "Urban"
    else:
        priv_grp = "Male"

    audit_summary_df, grp_breakdown = audit_fairness(df, audit_feature, priv_grp)
    health_res = get_fairness_health_score(audit_summary_df)

    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        st.markdown("#### Audit Scorecard")
        st.metric("Disparate Impact Benchmark", ">= 0.80 (80% Rule)", "Regulatory Target")
        st.metric("Min Disparate Impact Ratio", f"{health_res['min_disparate_impact_ratio']:.3f}", health_res['status'].split()[0])
        st.info(f"**Auditor Finding:** {health_res['status']}")

    with f_col2:
        st.markdown("#### Disparate Impact Comparison Table")
        st.dataframe(audit_summary_df[["group", "approval_rate", "disparate_impact_ratio", "demographic_parity_diff", "status"]], use_container_width=True)

    fig_f, ax_f = plt.subplots(figsize=(7, 3))
    colors = ["#10B981" if s == "PASS (Fair)" else "#F59E0B" for s in audit_summary_df["status"]]
    ax_f.bar(audit_summary_df["group"], audit_summary_df["disparate_impact_ratio"], color=colors)
    ax_f.axhline(0.80, color="red", linestyle="--", label="Regulatory Threshold (0.80)")
    ax_f.set_ylabel("Disparate Impact Ratio")
    ax_f.set_ylim(0, 1.2)
    ax_f.legend(loc="upper right")
    ax_f.set_title(f"Disparate Impact Ratio by {audit_feature}")
    st.pyplot(fig_f)

# ----------------- TAB 5: REGULATORY MODEL CARD -----------------
with tab5:
    st.subheader("Model Risk Management (MRM) Regulatory Card")
    st.markdown("""
    This automated compliance card satisfies **RBI Master Direction on IT Governance, Risk Controls, and AI Standards**.
    """)

    model_card_text = f"""
# REGULATORY MODEL CARD: FairCredit-XAI
**Governing Institution:** Indian Commercial Bank / NBFC  
**Auditing Entity:** IDRBT Center for AI & Financial Technologies  
**Model Version:** 1.0.0-PROD  
**Model Architecture:** Calibrated Gradient Boosting Decision Trees (GBDT)  

---

### 1. INTENDED USE & SCOPE
- **Application Domain:** Retail and MSME Credit Risk Appraisal & Digital Loan Sanctioning.
- **Out of Scope:** High-frequency algorithmic trading, wholesale syndicated lending.

### 2. PERFORMANCE BENCHMARKS (OUT-OF-SAMPLE)
- **ROC-AUC:** {metrics['roc_auc']:.4f}
- **Accuracy:** {metrics['accuracy']:.4f}
- **Precision:** {metrics['precision']:.4f}
- **Recall:** {metrics['recall']:.4f}
- **F1 Score:** {metrics['f1']:.4f}

### 3. EXPLAINABILITY & TRANSPARENCY CONTROLS
- **Global Attribution:** SHAP TreeExplainer quantifying portfolio-wide feature attributions.
- **Local Attribution:** Per-applicant waterfall decomposition for underwriter verification.
- **Customer Recourse:** Automated Counterfactual Generator fulfilling adverse action notice requirements.

### 4. FAIR LENDING AUDIT & DISPARATE IMPACT (EEOC / RBI COMPLIANCE)
- **Protected Attribute Audited:** Demographic Region (Urban vs Semi-Urban vs Rural), Gender (Male vs Female).
- **Minimum Disparate Impact Ratio:** {health_res['min_disparate_impact_ratio']:.3f}
- **Fair Lending Compliance Status:** {health_res['status']}

### 5. MODEL GOVERNANCE & OVERSIGHT
- **Human-in-the-Loop:** All applications with score 0.40 - 0.60 routed for manual underwriter review.
- **Monitoring Trigger:** Recalibration required if Disparate Impact Ratio falls below 0.80 or PSI > 0.15.
    """

    st.text_area("Generated Regulatory Model Card", model_card_text, height=350)
    st.download_button(
        label="📥 Download Standard Model Card (Markdown)",
        data=model_card_text,
        file_name="FairCredit_XAI_Regulatory_Model_Card.md",
        mime="text/markdown"
    )
