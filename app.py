"""
app.py
======
Workforce Attrition Patterns and Risk Hotspot Analysis
Interactive Streamlit dashboard for Palo Alto Networks HR leadership.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import load_and_prepare, attrition_rate

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Workforce Attrition & Risk Hotspot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")
PALETTE = {"No": "#2E7D32", "Yes": "#C62828"}

# ----------------------------------------------------------------------------
# DATA LOADING (cached)
# ----------------------------------------------------------------------------
@st.cache_data
def get_data():
    df, report = load_and_prepare("data/attrition.csv")
    return df, report

df_full, clean_report = get_data()

# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
st.sidebar.title("🔎 Filters")
st.sidebar.caption("All KPIs and charts update based on your selections below.")

departments = sorted(df_full["Department"].unique().tolist())
sel_departments = st.sidebar.multiselect("Department", departments, default=departments)

job_roles = sorted(df_full["JobRole"].unique().tolist())
sel_roles = st.sidebar.multiselect("Job Role", job_roles, default=job_roles)

min_tenure, max_tenure = int(df_full["YearsAtCompany"].min()), int(df_full["YearsAtCompany"].max())
sel_tenure = st.sidebar.slider("Years At Company (range)", min_tenure, max_tenure, (min_tenure, max_tenure))

overtime_filter = st.sidebar.radio("OverTime", ["All", "Yes", "No"], horizontal=True)

travel_filter = st.sidebar.selectbox(
    "Business Travel",
    ["All", "Non-Travel", "Travel_Rarely", "Travel_Frequently"]
)

st.sidebar.markdown("---")
if st.sidebar.button("Reset all filters"):
    st.rerun()

# ----------------------------------------------------------------------------
# APPLY FILTERS
# ----------------------------------------------------------------------------
df = df_full[
    df_full["Department"].isin(sel_departments) &
    df_full["JobRole"].isin(sel_roles) &
    df_full["YearsAtCompany"].between(sel_tenure[0], sel_tenure[1])
]
if overtime_filter != "All":
    df = df[df["OverTime"] == overtime_filter]
if travel_filter != "All":
    df = df[df["BusinessTravel"] == travel_filter]

if len(df) == 0:
    st.warning("No employees match the selected filters. Please broaden your selection.")
    st.stop()

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("📊 Workforce Attrition Patterns and Risk Hotspot Analysis")
st.caption("Palo Alto Networks — HR Analytics Dashboard")
st.markdown(
    f"Showing **{len(df):,}** of **{len(df_full):,}** employees based on current filters."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview",
    "🏢 Department & Role Heatmaps",
    "👥 Demographic Explorer",
    "⏳ Tenure & Workload",
])

# ============================================================================
# MODULE 1 — ATTRITION OVERVIEW
# ============================================================================
with tab1:
    st.header("Attrition Overview")

    total = len(df)
    left = int(df["AttritionFlag"].sum())
    retained = total - left
    rate = round(df["AttritionFlag"].mean() * 100, 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", f"{total:,}")
    c2.metric("Employees Who Left", f"{left:,}")
    c3.metric("Employees Retained", f"{retained:,}")
    c4.metric("Attrition Rate", f"{rate}%")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Retained vs Exited")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["AttritionLabel"].value_counts()
        colors = [PALETTE.get(k, "#999") for k in counts.index]
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
        ax.set_title("Employee Status Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with colB:
        st.subheader("Age Distribution by Attrition")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df, x="Age", hue="AttritionLabel", multiple="stack",
                     palette=PALETTE, bins=20, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "**How to read this:** the attrition rate is the share of employees in the "
        "currently filtered population who have left the organization. Use the sidebar "
        "filters to isolate specific departments, roles, tenure ranges, overtime, or "
        "travel patterns."
    )

# ============================================================================
# MODULE 2 — DEPARTMENT & ROLE HEATMAPS
# ============================================================================
with tab2:
    st.header("Department & Job Role Attrition")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Attrition Rate by Department")
        dept_stats = attrition_rate(df, "Department")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=dept_stats.index, y=dept_stats["AttritionRate(%)"],
                    ax=ax, color="#C62828")
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel("")
        ax.set_title("Department Attrition Rate")
        for i, v in enumerate(dept_stats["AttritionRate(%)"]):
            ax.text(i, v + 0.5, f"{v}%", ha="center")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(dept_stats, use_container_width=True)

    with colB:
        st.subheader("Attrition Rate by Job Role")
        role_stats = attrition_rate(df, "JobRole")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(y=role_stats.index, x=role_stats["AttritionRate(%)"],
                    ax=ax, color="#EF6C00")
        ax.set_xlabel("Attrition Rate (%)")
        ax.set_ylabel("")
        ax.set_title("Job Role Attrition Rate")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Heatmap — Department × Job Role Attrition Intensity")
    pivot = df.pivot_table(index="JobRole", columns="Department",
                            values="AttritionFlag", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax,
                cbar_kws={"label": "Attrition Rate (%)"})
    ax.set_title("Attrition Rate (%) by Job Role and Department")
    st.pyplot(fig)
    plt.close(fig)

    top_dept = dept_stats["AttritionRate(%)"].idxmax()
    top_role = role_stats["AttritionRate(%)"].idxmax()
    st.warning(
        f"**Highest-risk department:** {top_dept} "
        f"({dept_stats.loc[top_dept, 'AttritionRate(%)']}%) &nbsp;|&nbsp; "
        f"**Highest-risk role:** {top_role} "
        f"({role_stats.loc[top_role, 'AttritionRate(%)']}%)"
    )

# ============================================================================
# MODULE 3 — DEMOGRAPHIC ATTRITION EXPLORER
# ============================================================================
with tab3:
    st.header("Demographic Attrition Explorer")

    demo_choice = st.selectbox(
        "Choose a demographic dimension to explore",
        ["AgeGroup", "Gender", "Education", "EducationField", "MaritalStatus"]
    )

    demo_stats = attrition_rate(df, demo_choice)
    colA, colB = st.columns([2, 1])
    with colA:
        fig, ax = plt.subplots(figsize=(7, 4))
        order = demo_stats.index
        sns.barplot(x=demo_stats.index, y=demo_stats["AttritionRate(%)"],
                    order=order, ax=ax, color="#6A1B9A")
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel(demo_choice)
        ax.set_title(f"Attrition Rate by {demo_choice}")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
        plt.close(fig)
    with colB:
        st.dataframe(demo_stats, use_container_width=True)

    st.subheader("Age vs Job Satisfaction, colored by Attrition")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.stripplot(data=df, x="JobSatisfaction", y="Age", hue="AttritionLabel",
                  palette=PALETTE, dodge=True, alpha=0.6, ax=ax)
    ax.set_title("Age vs Job Satisfaction by Attrition Status")
    st.pyplot(fig)
    plt.close(fig)

# ============================================================================
# MODULE 4 — TENURE & WORKLOAD ANALYSIS
# ============================================================================
with tab4:
    st.header("Tenure & Workload Analysis")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Attrition by Tenure Bucket")
        tenure_stats = attrition_rate(df, "TenureGroup")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=tenure_stats.index, y=tenure_stats["AttritionRate(%)"],
                    ax=ax, color="#00838F")
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel("")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)
        plt.close(fig)

    with colB:
        st.subheader("Attrition by Promotion Gap")
        promo_stats = attrition_rate(df, "PromotionGapGroup")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=promo_stats.index, y=promo_stats["AttritionRate(%)"],
                    ax=ax, color="#AD1457")
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel("Years Since Last Promotion")
        st.pyplot(fig)
        plt.close(fig)

    colC, colD = st.columns(2)
    with colC:
        st.subheader("OverTime vs Attrition")
        ot_stats = attrition_rate(df, "OverTime")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=ot_stats.index, y=ot_stats["AttritionRate(%)"], ax=ax,
                    palette=["#C62828", "#2E7D32"])
        ax.set_ylabel("Attrition Rate (%)")
        st.pyplot(fig)
        plt.close(fig)

    with colD:
        st.subheader("Business Travel vs Attrition")
        bt_stats = attrition_rate(df, "BusinessTravel")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=bt_stats.index, y=bt_stats["AttritionRate(%)"], ax=ax,
                    color="#F9A825")
        ax.set_ylabel("Attrition Rate (%)")
        plt.xticks(rotation=15, ha="right")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Distance From Home vs Attrition")
    dist_stats = attrition_rate(df, "DistanceGroup")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=dist_stats.index, y=dist_stats["AttritionRate(%)"], ax=ax,
                color="#3949AB")
    ax.set_ylabel("Attrition Rate (%)")
    ax.set_xlabel("Distance From Home")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("🎯 Risk Hotspot Table")
    st.caption(
        "Segments ranked by attrition rate. Small-sample segments (fewer than 30 "
        "employees) are flagged so they are not mistaken for statistically robust hotspots."
    )

    def build_hotspots(data):
        rows = []
        specs = [
            ("Department", "Dept: "), ("JobRole", "Role: "),
            ("OverTime", "OverTime: "), ("BusinessTravel", "Travel: "),
            ("TenureGroup", "Tenure: "), ("AgeGroup", "Age: "),
            ("WorkloadRiskLabel", "Workload: "), ("PromotionGapGroup", "Promo Gap: "),
            ("MaritalStatus", "Marital: "),
        ]
        for col, prefix in specs:
            g = attrition_rate(data, col)
            for idx, row in g.iterrows():
                rows.append({
                    "Risk Hotspot": f"{prefix}{idx}",
                    "Attrition Rate (%)": row["AttritionRate(%)"],
                    "Employee Count": int(row["Employees"]),
                    "Small Sample (<30)": "⚠️ Yes" if row["Employees"] < 30 else "No",
                })
        out = pd.DataFrame(rows).sort_values("Attrition Rate (%)", ascending=False)
        return out.reset_index(drop=True)

    hotspots = build_hotspots(df)
    st.dataframe(hotspots.head(15), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# FOOTER — INSIGHTS & RECOMMENDATIONS QUICK ACCESS
# ----------------------------------------------------------------------------
st.markdown("---")
with st.expander("💡 Key Business Insights (auto-generated from filtered data)"):
    d = df
    st.markdown(f"""
- **Overall attrition rate** in the current selection is **{rate}%** ({left} of {total} employees).
- **Overtime employees** show an attrition rate of **{attrition_rate(d,'OverTime').loc['Yes','AttritionRate(%)'] if 'Yes' in attrition_rate(d,'OverTime').index else 'N/A'}%**
  versus non-overtime employees.
- The **highest-risk job role** in this selection is
  **{attrition_rate(d,'JobRole')['AttritionRate(%)'].idxmax()}**
  at **{attrition_rate(d,'JobRole')['AttritionRate(%)'].max()}%**.
- **Early-tenure employees (0-2 years)** show notably higher attrition than longer-tenured staff —
  a common pattern linked to onboarding, role fit, and early-career mobility.
""")

st.caption(
    "Data source: Palo Alto Networks HR dataset (1,470 employee records). "
    "Dashboard built with Streamlit, Pandas, Matplotlib, and Seaborn."
)
