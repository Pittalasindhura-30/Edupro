"""
data_utils.py
=============
Shared data loading, cleaning, and feature-engineering functions for the
Workforce Attrition Patterns and Risk Hotspot Analysis project.

Both the EDA notebook/script and the Streamlit dashboard (app.py) import
this module so that cleaning logic and derived features stay identical
everywhere they are used.
"""

import pandas as pd
import numpy as np


def load_raw_data(path="data/attrition.csv"):
    """Load the raw CSV exactly as provided."""
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """
    Clean the raw dataframe.

    Steps performed (see research paper Section 7 for full explanation):
    1. Drop exact duplicate rows (dataset has none, but this is kept for
       robustness in case the CSV changes).
    2. Standardize categorical text columns (strip whitespace).
    3. Confirm/standardize the Attrition column to a clean 0/1 integer
       ("AttritionFlag") plus a readable Yes/No column ("AttritionLabel").
    4. Validate numeric columns for negative/invalid values (none found
       in this dataset, but the checks are kept in for robustness).

    No missing values exist in this dataset (verified during inspection:
    df.isnull().sum().sum() == 0), so no imputation is performed. No
    outliers are removed — see the note in the research paper explaining
    why (e.g. long-tenure or high-income outliers are genuine, meaningful
    employees, not data errors, and removing them would bias attrition
    rates for those exact groups we most want to study).
    """
    df = df.copy()

    # 1. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)

    # 2. Standardize categorical text (strip stray whitespace/casing issues)
    cat_cols = ["BusinessTravel", "Department", "EducationField", "Gender",
                "JobRole", "MaritalStatus", "OverTime"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 3. Attrition standardization
    # In this dataset Attrition arrives already as 0/1 integers.
    # We keep that numeric flag and add a readable label column.
    if df["Attrition"].dtype == object:
        df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})
    else:
        df["AttritionFlag"] = df["Attrition"].astype(int)
    df["AttritionLabel"] = df["AttritionFlag"].map({1: "Yes", 0: "No"})

    # 4. Basic numeric sanity checks (no negative values should exist for
    # these inherently non-negative columns)
    non_negative_cols = ["Age", "DistanceFromHome", "MonthlyIncome",
                          "TotalWorkingYears", "YearsAtCompany",
                          "YearsInCurrentRole", "YearsSinceLastPromotion",
                          "YearsWithCurrManager", "NumCompaniesWorked"]
    invalid_counts = {}
    for c in non_negative_cols:
        if c in df.columns:
            n_invalid = (df[c] < 0).sum()
            if n_invalid > 0:
                invalid_counts[c] = int(n_invalid)
                df = df[df[c] >= 0]

    return df, {"duplicates_dropped": dropped, "invalid_negative_values": invalid_counts}


def engineer_features(df):
    """
    Add derived analytical features:
      - AgeGroup
      - TenureGroup (YearsAtCompany buckets)
      - CareerStage (based on TotalWorkingYears)
      - DistanceGroup (DistanceFromHome buckets)
      - PromotionGapGroup (YearsSinceLastPromotion buckets)
      - WorkloadRiskIndex (0-2 transparent score from OverTime + BusinessTravel)
    """
    df = df.copy()

    # --- Age Group ---
    age_bins = [0, 24, 34, 44, 54, np.inf]
    age_labels = ["Under 25", "25-34", "35-44", "45-54", "55+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=True)

    # --- Tenure Group (YearsAtCompany) ---
    tenure_bins = [-0.1, 2, 5, 10, 20, np.inf]
    tenure_labels = ["0-2 years", "3-5 years", "6-10 years", "11-20 years", "20+ years"]
    df["TenureGroup"] = pd.cut(df["YearsAtCompany"], bins=tenure_bins, labels=tenure_labels)

    # --- Career Stage (based on TotalWorkingYears, i.e. total career experience) ---
    def career_stage(years):
        if years <= 5:
            return "Early Career"
        elif years <= 15:
            return "Mid Career"
        else:
            return "Senior Career"
    df["CareerStage"] = df["TotalWorkingYears"].apply(career_stage)

    # --- Distance Group ---
    dist_bins = [-0.1, 5, 10, 20, np.inf]
    dist_labels = ["0-5 km", "6-10 km", "11-20 km", "21+ km"]
    df["DistanceGroup"] = pd.cut(df["DistanceFromHome"], bins=dist_bins, labels=dist_labels)

    # --- Promotion Gap Group ---
    promo_bins = [-0.1, 1, 3, 7, np.inf]
    promo_labels = ["0-1 years", "2-3 years", "4-7 years", "8+ years"]
    df["PromotionGapGroup"] = pd.cut(df["YearsSinceLastPromotion"], bins=promo_bins, labels=promo_labels)

    # --- Workload Risk Index ---
    # Transparent 0-2 score: +1 if OverTime == Yes, +1 if BusinessTravel == Travel_Frequently
    # Assumption: frequent overtime and frequent travel are the two workload/mobility
    # factors most consistently linked to burnout in the literature and available here.
    # 0 = Low workload risk, 1 = Moderate, 2 = High
    df["WorkloadRiskIndex"] = (
        (df["OverTime"] == "Yes").astype(int) +
        (df["BusinessTravel"] == "Travel_Frequently").astype(int)
    )
    workload_labels = {0: "Low", 1: "Moderate", 2: "High"}
    df["WorkloadRiskLabel"] = df["WorkloadRiskIndex"].map(workload_labels)

    return df


def load_and_prepare(path="data/attrition.csv"):
    """Convenience function: load -> clean -> engineer, returns (df, clean_report)."""
    raw = load_raw_data(path)
    clean, report = clean_data(raw)
    final = engineer_features(clean)
    return final, report


def attrition_rate(df, group_col=None):
    """
    Return overall attrition rate (%) if group_col is None,
    otherwise a DataFrame of attrition rate, headcount, and leavers per group.
    """
    if group_col is None:
        return round(df["AttritionFlag"].mean() * 100, 2)

    g = df.groupby(group_col, observed=True).agg(
        Employees=("AttritionFlag", "size"),
        Left=("AttritionFlag", "sum"),
    )
    g["AttritionRate(%)"] = (g["Left"] / g["Employees"] * 100).round(2)
    g["Retained"] = g["Employees"] - g["Left"]
    return g.sort_values("AttritionRate(%)", ascending=False)


if __name__ == "__main__":
    df, report = load_and_prepare()
    print("Cleaning report:", report)
    print("Shape:", df.shape)
    print("Overall attrition rate:", attrition_rate(df), "%")
