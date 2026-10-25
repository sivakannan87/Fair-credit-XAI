"""
Fairness & Bias Auditing Module for Responsible AI in Banking.
Evaluates compliance against Fair Lending, Disparate Impact (4/5ths Rule),
Demographic Parity, and Equal Opportunity across demographic sub-groups.
"""

import pandas as pd
import numpy as np

def compute_group_metrics(df: pd.DataFrame, group_col: str, pred_col: str = "pred_approved", true_col: str = "approved"):
    """
    Computes approval rates, accuracy, TPR (Equal Opportunity) per demographic sub-group.
    """
    groups = df[group_col].unique()
    results = []

    for g in groups:
        sub = df[df[group_col] == g]
        total = len(sub)
        approvals = sub[pred_col].sum()
        approval_rate = approvals / total if total > 0 else 0.0

        # TPR (True Positive Rate) = Approvals among actually creditworthy applicants
        creditworthy = sub[sub[true_col] == 1]
        tpr = creditworthy[pred_col].sum() / len(creditworthy) if len(creditworthy) > 0 else 0.0

        results.append({
            "group": g,
            "sample_count": total,
            "approved_count": int(approvals),
            "approval_rate": round(approval_rate, 4),
            "true_positive_rate": round(tpr, 4)
        })

    return pd.DataFrame(results)

def audit_fairness(df: pd.DataFrame, protected_attribute: str, privileged_group: str, pred_col: str = "pred_approved", true_col: str = "approved"):
    """
    Calculates Disparate Impact Ratio, Demographic Parity Difference, and Equal Opportunity Difference.
    """
    group_df = compute_group_metrics(df, protected_attribute, pred_col, true_col)
    
    priv_row = group_df[group_df["group"] == privileged_group]
    if priv_row.empty:
        raise ValueError(f"Privileged group '{privileged_group}' not found in column '{protected_attribute}'.")
    
    priv_rate = priv_row["approval_rate"].values[0]
    priv_tpr = priv_row["true_positive_rate"].values[0]

    audit_summary = []
    for _, row in group_df.iterrows():
        g = row["group"]
        rate = row["approval_rate"]
        tpr = row["true_positive_rate"]

        # Disparate Impact Ratio (DIR) = Approval Rate(Unprivileged) / Approval Rate(Privileged)
        dir_val = rate / priv_rate if priv_rate > 0 else 1.0

        # Demographic Parity Difference = Approval Rate(Unprivileged) - Approval Rate(Privileged)
        dpd_val = rate - priv_rate

        # Equal Opportunity Difference = TPR(Unprivileged) - TPR(Privileged)
        eod_val = tpr - priv_tpr

        # Compliance Check (US EEOC 4/5ths rule & RBI fair lending standard: DIR >= 0.80)
        compliant = dir_val >= 0.80

        audit_summary.append({
            "attribute": protected_attribute,
            "group": g,
            "is_privileged": (g == privileged_group),
            "approval_rate": rate,
            "disparate_impact_ratio": round(dir_val, 3),
            "demographic_parity_diff": round(dpd_val, 3),
            "equal_opportunity_diff": round(eod_val, 3),
            "four_fifths_compliant": compliant,
            "status": "PASS (Fair)" if compliant else "ALERT (Potential Bias)"
        })

    return pd.DataFrame(audit_summary), group_df

def get_fairness_health_score(audit_df: pd.DataFrame) -> dict:
    """
    Calculates a single composite fairness health index (0 - 100%) for executive presentation.
    """
    dirs = audit_df[~audit_df["is_privileged"]]["disparate_impact_ratio"].values
    if len(dirs) == 0:
        return {"health_score": 100.0, "status": "Compliant"}
    
    min_dir = min(dirs)
    # Scaled where DIR >= 0.80 is > 90% score
    if min_dir >= 0.80:
        score = min(100.0, 80 + (min_dir - 0.8) * 100)
        status = "COMPLIANT (Meets RBI Fair Lending Benchmarks)"
    else:
        score = max(30.0, min_dir * 100)
        status = "WARNING (Disparate Impact Detected - Requires Mitigation)"

    return {
        "health_score": round(score, 1),
        "min_disparate_impact_ratio": round(min_dir, 3),
        "status": status
    }
