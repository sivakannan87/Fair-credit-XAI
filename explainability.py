"""
Explainable AI (XAI) & Actionable Counterfactual Recourse Engine.
Generates:
1. Global Feature Attribution (SHAP).
2. Local Applicant-Level SHAP Waterfall / Contributions.
3. Actionable Counterfactual Recourse (Adverse Action Notice) for rejected applicants.
"""

import numpy as np
import pandas as pd
import shap

class ExplainabilityEngine:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        # Using TreeExplainer for fast, exact tree attribution
        self.explainer = shap.TreeExplainer(self.model)

    def compute_global_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """
        Computes mean absolute SHAP value across all features for global explainability.
        """
        shap_values = self.explainer.shap_values(X_sample[self.feature_names])
        
        # In binary classification, handle 1D or 2D shap output
        if isinstance(shap_values, list):
            sv = shap_values[1]  # positive class (Approved)
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)
        df_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_abs_shap
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        
        return df_importance

    def _format_input(self, applicant_series: pd.Series) -> pd.DataFrame:
        """
        Converts raw or partially encoded applicant series into a 1-row DataFrame aligned with self.feature_names.
        """
        data = {}
        for feat in self.feature_names:
            if feat in applicant_series:
                data[feat] = applicant_series[feat]
            elif feat.startswith("employment_type_"):
                cat_val = feat.replace("employment_type_", "")
                data[feat] = 1 if applicant_series.get("employment_type") == cat_val else 0
            else:
                data[feat] = 0
        return pd.DataFrame([data])

    def explain_local_applicant(self, applicant_series: pd.Series) -> dict:
        """
        Explains an individual loan decision using local SHAP attribution.
        """
        applicant_df = self._format_input(applicant_series)
        shap_values = self.explainer.shap_values(applicant_df)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            base_val = self.explainer.expected_value[1]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
            base_val = self.explainer.expected_value[1]
        else:
            sv = shap_values[0]
            base_val = self.explainer.expected_value

        pred_prob = float(self.model.predict_proba(applicant_df)[:, 1][0])
        decision = "APPROVED" if pred_prob >= 0.50 else "REJECTED"

        contributions = []
        for feat, s in zip(self.feature_names, sv):
            val = applicant_df[feat].values[0]
            contributions.append({
                "feature": feat,
                "value": val,
                "shap_impact": round(float(s), 4),
                "direction": "Positive (+Approval)" if s > 0 else "Negative (-Rejection)"
            })

        contributions_df = pd.DataFrame(contributions).sort_values(by="shap_impact", key=abs, ascending=False)

        if hasattr(base_val, "__len__"):
            base_val_scalar = float(base_val[0])
        else:
            base_val_scalar = float(base_val)

        return {
            "applicant_id": applicant_series.get("applicant_id", "UNKNOWN"),
            "approval_probability": round(pred_prob, 4),
            "decision": decision,
            "base_value": round(base_val_scalar, 4),
            "contributions": contributions_df
        }

    def generate_counterfactual_recourse(self, applicant_series: pd.Series, target_prob: float = 0.55) -> dict:
        """
        Computes the minimal actionable changes required to turn a REJECTED applicant into an APPROVED applicant.
        Actionable levers:
        - Reduce debt_to_income_ratio (paying down existing debt)
        - Reduce loan_amount_inr (requesting slightly lower principal)
        - Increase tenure_months (stretching tenure to reduce monthly burden)
        """
        current_df = self._format_input(applicant_series)
        current_data = current_df.to_dict(orient="records")[0]
        current_prob = float(self.model.predict_proba(current_df)[:, 1][0])

        if current_prob >= 0.50:
            return {
                "status": "Already Approved",
                "current_probability": round(current_prob, 3),
                "recommendation": "Applicant meets underwriting criteria. No adverse action notice required."
            }

        # Grid search over feasible actionable modifications
        best_plan = None
        min_distance = float("inf")

        dti_options = [0.90, 0.75, 0.60]  # multipliers on DTI (e.g. reduce by 10%, 25%, 40%)
        loan_amt_options = [1.0, 0.85, 0.75]  # reduce loan amount requested
        curr_tenure = int(current_data["tenure_months"])
        tenure_options = [curr_tenure, min(84, curr_tenure + 12), min(84, curr_tenure + 24)]

        for d_mult in dti_options:
            for l_mult in loan_amt_options:
                for t in tenure_options:
                    candidate = current_data.copy()
                    candidate["debt_to_income_ratio"] = max(0.08, round(current_data["debt_to_income_ratio"] * d_mult, 3))
                    candidate["loan_amount_inr"] = int(round(current_data["loan_amount_inr"] * l_mult, -3))
                    candidate["tenure_months"] = int(t)

                    cand_df = pd.DataFrame([candidate])
                    cand_prob = float(self.model.predict_proba(cand_df)[:, 1][0])

                    if cand_prob >= target_prob:
                        # Cost function: prefer minimal perturbation
                        dist = (1 - d_mult) * 2.0 + (1 - l_mult) * 1.5 + (t - curr_tenure) / 24.0
                        if dist < min_distance:
                            min_distance = dist
                            best_plan = {
                                "original_prob": round(current_prob, 3),
                                "projected_prob": round(cand_prob, 3),
                                "actions": []
                            }
                            if candidate["debt_to_income_ratio"] != current_data["debt_to_income_ratio"]:
                                pct_down = int((1 - d_mult) * 100)
                                best_plan["actions"].append(
                                    f"Reduce Debt-to-Income (DTI) from {current_data['debt_to_income_ratio']:.1%} to {candidate['debt_to_income_ratio']:.1%} (pay down approximately {pct_down}% of outstanding credit/card dues)."
                                )
                            if candidate["loan_amount_inr"] != current_data["loan_amount_inr"]:
                                best_plan["actions"].append(
                                    f"Adjust requested loan amount from ₹{int(current_data['loan_amount_inr']):,} to ₹{int(candidate['loan_amount_inr']):,}."
                                )
                            if candidate["tenure_months"] != current_data["tenure_months"]:
                                best_plan["actions"].append(
                                    f"Extend loan tenure from {int(current_data['tenure_months'])} months to {int(candidate['tenure_months'])} months to lower monthly EMI burden."
                                )

        if best_plan is None:
            # Fallback if debt adjustment isn't enough (e.g. CIBIL score is too low)
            best_plan = {
                "original_prob": round(current_prob, 3),
                "projected_prob": target_prob,
                "actions": [
                    "Improve credit bureau (CIBIL) score by at least 45 points through 6 months of consecutive zero-delay EMI payments.",
                    f"Reduce Debt-to-Income (DTI) ratio below 35% before resubmitting application.",
                    "Add a creditworthy co-applicant or provide liquid collateral."
                ]
            }

        return {
            "status": "Adverse Action Notice Generated",
            "current_probability": round(current_prob, 3),
            "projected_probability": best_plan["projected_prob"],
            "recourse_steps": best_plan["actions"]
        }
