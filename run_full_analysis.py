"""
run_full_analysis.py
=====================
Runs the complete exploratory data analysis and prints every statistic
needed for the KPIs, risk hotspot table, insights, and research paper.
All numbers below are computed directly from the dataset -- nothing is
invented. Output is captured and used verbatim in later deliverables.
"""
import pandas as pd
import numpy as np
from data_utils import load_and_prepare, attrition_rate

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

df, report = load_and_prepare()

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

section("1. DATASET SHAPE & CLEANING REPORT")
print("Rows x Cols:", df.shape)
print("Cleaning report:", report)
print("Missing values total:", df.isnull().sum().sum())

section("2. OVERALL ATTRITION")
total = len(df)
left = df['AttritionFlag'].sum()
retained = total - left
rate = attrition_rate(df)
print(f"Total employees: {total}")
print(f"Left: {left}")
print(f"Retained: {retained}")
print(f"Attrition rate: {rate}%")
print(f"Retention rate: {round(100-rate,2)}%")

section("3. DEPARTMENT-WISE ATTRITION")
dept = attrition_rate(df, "Department")
print(dept)

section("4. JOB ROLE ATTRITION")
role = attrition_rate(df, "JobRole")
print(role)

section("5. AGE GROUP ATTRITION")
age = attrition_rate(df, "AgeGroup")
print(age)

section("6. GENDER ATTRITION")
gender = attrition_rate(df, "Gender")
print(gender)

section("7. MARITAL STATUS ATTRITION")
marital = attrition_rate(df, "MaritalStatus")
print(marital)

section("8. EDUCATION LEVEL ATTRITION")
edu = attrition_rate(df, "Education")
print(edu)

section("9. EDUCATION FIELD ATTRITION")
edufield = attrition_rate(df, "EducationField")
print(edufield)

section("10. SATISFACTION / INVOLVEMENT FACTORS")
for col in ["EnvironmentSatisfaction", "JobSatisfaction", "JobInvolvement",
            "RelationshipSatisfaction", "WorkLifeBalance"]:
    print(f"\n--- {col} ---")
    print(attrition_rate(df, col))

section("11. TENURE & CAREER STAGE")
print("\n--- TenureGroup (YearsAtCompany) ---")
print(attrition_rate(df, "TenureGroup"))
print("\n--- CareerStage ---")
print(attrition_rate(df, "CareerStage"))
print("\n--- PromotionGapGroup (YearsSinceLastPromotion) ---")
print(attrition_rate(df, "PromotionGapGroup"))

section("12. WORKLOAD & MOBILITY")
print("\n--- OverTime ---")
print(attrition_rate(df, "OverTime"))
print("\n--- BusinessTravel ---")
print(attrition_rate(df, "BusinessTravel"))
print("\n--- DistanceGroup ---")
print(attrition_rate(df, "DistanceGroup"))
print("\n--- WorkloadRiskLabel ---")
print(attrition_rate(df, "WorkloadRiskLabel"))

section("13. COMPENSATION")
print("\nMonthlyIncome by Attrition:")
print(df.groupby("AttritionLabel")["MonthlyIncome"].describe()[["mean","50%","std"]])
print("\nJobLevel attrition:")
print(attrition_rate(df, "JobLevel"))
print("\nStockOptionLevel attrition:")
print(attrition_rate(df, "StockOptionLevel"))
print("\nPercentSalaryHike by Attrition:")
print(df.groupby("AttritionLabel")["PercentSalaryHike"].mean())

section("14. CORRELATION WITH ATTRITION (numeric cols, point-biserial via corr)")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != "Attrition"]
corr = df[num_cols].corr()["AttritionFlag"].sort_values(ascending=False)
print(corr)

section("15. KPI CALCULATIONS")
print(f"KPI1 Overall Attrition Rate: {rate}%")
print("\nKPI2 Department Attrition Rate:")
print(dept["AttritionRate(%)"])
print("\nKPI3 Role Attrition Rate:")
print(role["AttritionRate(%)"])

early_tenure = df[df["YearsAtCompany"] <= 2]
early_rate = round(early_tenure["AttritionFlag"].mean()*100, 2)
print(f"\nKPI4 Early-Tenure Attrition (YearsAtCompany <= 2): {early_rate}% "
      f"(n={len(early_tenure)}, left={early_tenure['AttritionFlag'].sum()})")

print("\nKPI5 Workload Attrition Index (WorkloadRiskLabel):")
print(attrition_rate(df, "WorkloadRiskLabel"))

section("16. RISK HOTSPOT CANDIDATES (rate, count, left)")
hotspot_candidates = []
def add_candidates(col, label_prefix=""):
    g = attrition_rate(df, col)
    for idx, row in g.iterrows():
        hotspot_candidates.append({
            "Segment": f"{label_prefix}{idx}",
            "AttritionRate": row["AttritionRate(%)"],
            "Employees": row["Employees"],
            "Left": row["Left"]
        })

add_candidates("Department", "Dept: ")
add_candidates("JobRole", "Role: ")
add_candidates("OverTime", "OverTime: ")
add_candidates("BusinessTravel", "Travel: ")
add_candidates("TenureGroup", "Tenure: ")
add_candidates("AgeGroup", "Age: ")
add_candidates("JobSatisfaction", "JobSat=")
add_candidates("WorkloadRiskLabel", "Workload: ")
add_candidates("PromotionGapGroup", "PromoGap: ")
add_candidates("MaritalStatus", "Marital: ")
add_candidates("StockOptionLevel", "StockOpt=")

hs_df = pd.DataFrame(hotspot_candidates).sort_values("AttritionRate", ascending=False)
print(hs_df.to_string(index=False))

section("17. CROSS-TAB: OverTime x JobRole (attrition rate %) -- interesting intersection")
ct = df.pivot_table(index="JobRole", columns="OverTime", values="AttritionFlag", aggfunc="mean") * 100
print(ct.round(2))

section("18. Sales Executive / Sales Rep + Overtime deep dive")
sr = df[(df["JobRole"]=="Sales Representative")]
print("Sales Representative overall n:", len(sr), "attrition rate:", round(sr["AttritionFlag"].mean()*100,2))
sr_ot = df[(df["JobRole"]=="Sales Representative") & (df["OverTime"]=="Yes")]
print("Sales Rep + OT=Yes n:", len(sr_ot), "attrition rate:", round(sr_ot["AttritionFlag"].mean()*100,2) if len(sr_ot)>0 else "n/a")

section("19. Single + Early tenure")
single_early = df[(df["MaritalStatus"]=="Single") & (df["YearsAtCompany"]<=2)]
print("Single & YearsAtCompany<=2 n:", len(single_early), "rate:", round(single_early["AttritionFlag"].mean()*100,2))

print("\n\nDONE")
