"""
Synthetic Indian Banking Credit Dataset Generator.
Simulates realistic retail and MSME loan applicants with key financial parameters,
credit bureau metrics (CIBIL score), and sensitive demographic attributes for fairness auditing.
"""

import numpy as np
import pandas as pd

def generate_credit_data(n_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    # 1. Demographic Features (Sensitive & Protected Attributes)
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.65, 0.35])
    demographic_region = np.random.choice(["Urban", "Semi-Urban", "Rural"], size=n_samples, p=[0.50, 0.30, 0.20])
    age = np.random.randint(21, 62, size=n_samples)
    employment_type = np.random.choice(["Salaried", "Self-Employed", "Gig Economy"], size=n_samples, p=[0.55, 0.35, 0.10])

    # 2. Financial & Credit Features (Indian context)
    # Base monthly income in INR
    base_income = {
        "Urban": np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples),       # ~INR 45k - 100k
        "Semi-Urban": np.random.lognormal(mean=10.4, sigma=0.45, size=n_samples), # ~INR 30k - 65k
        "Rural": np.random.lognormal(mean=10.0, sigma=0.4, size=n_samples)        # ~INR 20k - 45k
    }
    monthly_income_inr = np.array([
        base_income[region][i] for i, region in enumerate(demographic_region)
    ])
    monthly_income_inr = np.clip(np.round(monthly_income_inr, -2), 15000, 350000)

    # CIBIL Score (300 - 900)
    cibil_score = np.random.normal(loc=710, scale=80, size=n_samples)
    cibil_score = np.clip(np.round(cibil_score), 300, 900).astype(int)

    # Existing loans count
    existing_loans_count = np.random.poisson(lam=1.5, size=n_samples)
    existing_loans_count = np.clip(existing_loans_count, 0, 7)

    # Requested loan amount in INR (typically 2x to 15x monthly income)
    income_multiplier = np.random.uniform(3, 12, size=n_samples)
    loan_amount_inr = np.round(monthly_income_inr * income_multiplier, -3)
    loan_amount_inr = np.clip(loan_amount_inr, 50000, 2500000)

    # Loan tenure in months (12 to 84 months)
    tenure_months = np.random.choice([12, 24, 36, 48, 60, 72, 84], size=n_samples, p=[0.1, 0.2, 0.3, 0.15, 0.15, 0.05, 0.05])

    # Debt to Income Ratio (DTI) - monthly existing obligations / gross monthly income
    debt_to_income_ratio = np.random.beta(a=2, b=4, size=n_samples)
    debt_to_income_ratio = np.clip(np.round(debt_to_income_ratio, 3), 0.05, 0.85)

    # Account Aggregator (AA) data score / digital transaction health (0 - 100)
    digital_banking_score = np.random.randint(30, 99, size=n_samples)

    # 3. Ground Truth Approval Decision Logic (Reflecting credit risk with potential systemic disparities)
    # Real-world scoring formula (logit model)
    z = (
        0.015 * (cibil_score - 650)
        + 0.000025 * (monthly_income_inr - 40000)
        - 4.5 * (debt_to_income_ratio - 0.35)
        - 0.0000015 * (loan_amount_inr / (tenure_months + 1))
        - 0.25 * existing_loans_count
        + 0.02 * (digital_banking_score - 50)
    )

    # Sigmoid function for probability of approval
    prob_approval = 1 / (1 + np.exp(-z))

    # Binary outcome: 1 = Approved / Low Risk, 0 = Rejected / High Default Risk
    approved = (prob_approval >= 0.50).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"APP-IN-{10000 + i}" for i in range(n_samples)],
        "gender": gender,
        "demographic_region": demographic_region,
        "age": age,
        "employment_type": employment_type,
        "monthly_income_inr": monthly_income_inr.astype(int),
        "cibil_score": cibil_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "loan_amount_inr": loan_amount_inr.astype(int),
        "tenure_months": tenure_months,
        "existing_loans_count": existing_loans_count,
        "digital_banking_score": digital_banking_score,
        "approved": approved
    })

    return df

if __name__ == "__main__":
    df = generate_credit_data(2500)
    df.to_csv("d:/IDRBT banking project/loan_data.csv", index=False)
    print(f"Generated {len(df)} loan applications. Approval rate: {df['approved'].mean():.2%}")
    print(df.head())
