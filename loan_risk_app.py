# ============================================================
# LOAN RISK ANALYTICS — STREAMLIT WEB APP
# Run:  streamlit run loan_risk_app.py
# Keep EXTERNAL_CIBIL_DATA.xlsx & INTERNAL_BANK_DATA.xlsx
# in the same folder as this file
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings("ignore")

# ── colours matching the Power BI dashboard ──────────────────
COLOURS = {"P1":"#1a6e3c", "P2":"#2196F3", "P3":"#FF9800", "P4":"#e53935"}

# ── features used for the ML model ───────────────────────────
FEATURES = [
    "num_times_delinquent","num_times_30p_dpd","num_times_60p_dpd",
    "num_deliq_6mts","num_deliq_12mts",
    "num_sub","num_dbt","num_lss","num_std",
    "tot_enq","enq_L6m","enq_L3m",
    "AGE","NETMONTHLYINCOME","Time_With_Curr_Empr",
    "CC_Flag","PL_Flag","HL_Flag","GL_Flag",
    "Total_TL","Tot_Active_TL","Tot_Missed_Pmnt",
    "Age_Oldest_TL","Age_Newest_TL",
    "CC_utilization","PL_utilization","Credit_Score",
]

# ============================================================
# STEP 1 — LOAD & CLEAN DATA
# ============================================================
@st.cache_data(show_spinner="Loading data...")
def load_data():
    cibil = pd.read_excel("EXTERNAL_CIBIL_DATA.xlsx")
    bank  = pd.read_excel("INTERNAL_BANK_DATA.xlsx")
    df    = cibil.merge(bank, on="PROSPECTID")

    # -99999 means "no data" in this dataset — replace with NaN
    df = df.replace(-99999, np.nan)

    # Fix outliers
    if "pct_currentBal_all_TL" in df.columns:
        df["pct_currentBal_all_TL"] = df["pct_currentBal_all_TL"].clip(upper=100)
    df["NETMONTHLYINCOME"] = df["NETMONTHLYINCOME"].replace(0, np.nan)

    # New useful columns
    df["Is_High_Risk"]  = df["Approved_Flag"].isin(["P3","P4"]).astype(int)
    df["Risk_Category"] = df["Approved_Flag"].map(
                             {"P1":"Low Risk","P2":"Medium Risk",
                              "P3":"High Risk","P4":"High Risk"})
    df["Ever_Delinquent"] = df["max_delinquency_level"].notna().astype(int)

    df["Credit_Score_Band"] = pd.cut(df["Credit_Score"],
        bins=[0,599,649,699,749,900],
        labels=["Very Poor","Poor","Fair","Good","Excellent"])

    df["Age_Group"] = pd.cut(df["AGE"],
        bins=[17,25,35,50,100],
        labels=["18-25","26-35","36-50","50+"])

    df["Income_Band"] = pd.cut(df["NETMONTHLYINCOME"].fillna(-1),
        bins=[-2,0,15000,30000,60000,1e8],
        labels=["Unknown","Low <15K","Mid 15-30K","Upper 30-60K","High >60K"])

    df["Missed_Pmnt_Band"] = pd.cut(df["Tot_Missed_Pmnt"],
        bins=[-1,0,2,5,1000],
        labels=["None","1-2","3-5","6+"])
    return df


# ============================================================
# STEP 2 — TRAIN MODEL
# ============================================================
@st.cache_resource(show_spinner="Training model...")
def train_model(df):
    data = df[FEATURES + ["Is_High_Risk"]].dropna()
    X, y = data[FEATURES], data["Is_High_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)

    model  = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 C=0.1, random_state=42)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    importance = pd.DataFrame({"Feature": FEATURES,
                                "Importance": np.abs(model.coef_[0])
                               }).sort_values("Importance", ascending=False).head(15)
    return model, scaler, y_test, y_pred, y_prob, importance


# ============================================================
# STEP 3 — APP
# ============================================================
def main():
    st.set_page_config(page_title="Loan Risk Analytics",
                       page_icon="🏦", layout="wide")

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=55)
    st.sidebar.markdown("## 🏦 Loan Risk Analytics")
    st.sidebar.divider()
    page = st.sidebar.selectbox("Navigate", [
        "🏠 Overview",
        "📊 Descriptive Analytics",
        "🔍 Diagnostic Analytics",
        "🤖 Predictive Model",
        "🎯 Risk Scorer"
    ])
    st.sidebar.caption("51,336 Customers | 88 Features | 2026")

    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Excel files not found. Place both .xlsx files in the same folder.")
        st.stop()

    # ── PAGE 1: OVERVIEW ─────────────────────────────────────
    if page == "🏠 Overview":
        st.title("🏦 Loan Risk Analytics Dashboard")
        st.caption("Data-Powered Lending Decisions | Indian Retail Banking | 2026")
        st.divider()

        # KPI cards
        total   = len(df)
        hi_risk = df["Is_High_Risk"].sum()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Customers",    f"{total:,}")
        c2.metric("High Risk (P3+P4)",  f"{hi_risk:,}", f"{hi_risk/total*100:.1f}%")
        c3.metric("Avg Credit Score",   f"{df['Credit_Score'].mean():.2f}")
        c4.metric("Avg Monthly Income", f"₹{df['NETMONTHLYINCOME'].mean():,.0f}")

        st.divider()
        col1, col2 = st.columns(2)

        # Donut — approval tier
        with col1:
            counts = df["Approved_Flag"].value_counts().reset_index()
            counts.columns = ["Tier","Count"]
            counts["Pct"]  = (counts["Count"]/total*100).round(2)
            fig = px.pie(counts, values="Count", names="Tier", hole=0.45,
                         color="Tier", color_discrete_map=COLOURS,
                         title="Count of Customers by Approved_Flag")
            fig.update_traces(textinfo="label+percent")
            st.plotly_chart(fig, use_container_width=True)

        # Avg credit score by education
        with col2:
            edu = df.groupby("EDUCATION")["Credit_Score"].mean().reset_index()
            edu.columns = ["Education","Avg Credit Score"]
            edu = edu.sort_values("Avg Credit Score", ascending=False)
            fig = px.bar(edu, x="Avg Credit Score", y="Education",
                         orientation="h", text="Avg Credit Score",
                         title="Avg Credit Score by EDUCATION",
                         color="Avg Credit Score",
                         color_continuous_scale="Blues")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(plot_bgcolor="white", showlegend=False,
                              xaxis_range=[676,686])
            st.plotly_chart(fig, use_container_width=True)

        col3, col4, col5 = st.columns(3)

        # Gender pie
        with col3:
            fig = px.pie(df, names="GENDER", hole=0.4,
                         title="Count by GENDER",
                         color="GENDER",
                         color_discrete_sequence=["#2196F3","#FF9800"])
            st.plotly_chart(fig, use_container_width=True)

        # Risk category bar
        with col4:
            rc = df["Risk_Category"].value_counts().reset_index()
            rc.columns = ["Risk Category","Count"]
            fig = px.bar(rc, x="Risk Category", y="Count",
                         color="Risk Category",
                         color_discrete_map={"Low Risk":"#1a6e3c",
                                             "Medium Risk":"#2196F3",
                                             "High Risk":"#e53935"},
                         title="Count by Risk Category", text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(plot_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Income vs credit score scatter
        with col5:
            sample = df.sample(1500, random_state=1)
            fig = px.scatter(sample, x="Credit_Score", y="NETMONTHLYINCOME",
                             color="Approved_Flag", color_discrete_map=COLOURS,
                             opacity=0.5, title="Income vs Credit Score",
                             labels={"NETMONTHLYINCOME":"Income"})
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        st.info("**P1** = Low Risk | **P2** = Medium Risk | "
                "**P3** = Elevated | **P4** = High Risk")

    # ── PAGE 2: DESCRIPTIVE ───────────────────────────────────
    elif page == "📊 Descriptive Analytics":
        st.title("📊 Descriptive Analytics")
        st.caption("Who are our customers?")
        st.divider()

        col1, col2 = st.columns(2)

        # Age group bar
        with col1:
            ag = df.groupby("Age_Group", observed=True).size().reset_index(name="Count")
            fig = px.bar(ag, x="Age_Group", y="Count", color="Age_Group",
                         title="Age Group-wise Customer Distribution", text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(plot_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Gender × Approved_Flag stacked bar
        with col2:
            gd = df.groupby(["GENDER","Approved_Flag"]).size().reset_index(name="Count")
            fig = px.bar(gd, x="GENDER", y="Count", color="Approved_Flag",
                         color_discrete_map=COLOURS, barmode="stack",
                         title="Loan Approval Distribution by Gender", text="Count")
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        # Education × Approved_Flag 100% stacked
        with col3:
            ed = df.groupby(["EDUCATION","Approved_Flag"]).size().reset_index(name="Count")
            fig = px.bar(ed, x="Count", y="EDUCATION", color="Approved_Flag",
                         color_discrete_map=COLOURS, barmode="relative",
                         orientation="h",
                         title="Loan Approval by Education Level (%)")
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # Age group risk distribution
        with col4:
            ar = df.groupby(["Age_Group","Risk_Category"],
                            observed=True).size().reset_index(name="Count")
            fig = px.bar(ar, x="Age_Group", y="Count", color="Risk_Category",
                         color_discrete_map={"Low Risk":"#1a6e3c",
                                             "Medium Risk":"#2196F3",
                                             "High Risk":"#e53935"},
                         title="Age Group-wise Risk Distribution",
                         text="Count", barmode="group")
            fig.update_traces(textposition="outside")
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # Credit score band summary table
        st.divider()
        st.markdown("**Credit Score Range Distribution**")
        cs = df.groupby("Credit_Score_Band", observed=True).agg(
            Count=("PROSPECTID","count"),
            High_Risk_Pct=("Is_High_Risk","mean"),
            Avg_Income=("NETMONTHLYINCOME","mean")
        ).reset_index()
        cs["High_Risk_Pct"] = (cs["High_Risk_Pct"]*100).round(1)
        cs["Avg_Income"]    = cs["Avg_Income"].round(0)
        st.dataframe(cs, use_container_width=True)

    # ── PAGE 3: DIAGNOSTIC ────────────────────────────────────
    elif page == "🔍 Diagnostic Analytics":
        st.title("🔍 Diagnostic Analytics")
        st.caption("Why do customers default?")
        st.divider()

        col1, col2 = st.columns(2)

        # Approval by risk category
        with col1:
            ra = df.groupby(["Risk_Category","Approved_Flag"]).size().reset_index(name="Count")
            fig = px.bar(ra, x="Count", y="Risk_Category", color="Approved_Flag",
                         color_discrete_map=COLOURS, barmode="relative",
                         orientation="h", title="Approval by Risk Category (%)")
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # Credit score band distribution
        with col2:
            cs = df["Credit_Score_Band"].value_counts().reset_index()
            cs.columns = ["Band","Count"]
            fig = px.bar(cs, x="Band", y="Count", color="Band",
                         title="Credit Score Range Distribution", text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(plot_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        # Missed payments vs risk rate
        with col3:
            mp = df.groupby("Missed_Pmnt_Band",
                            observed=True)["Is_High_Risk"].mean().reset_index()
            mp["High_Risk_%"] = (mp["Is_High_Risk"]*100).round(1)
            fig = px.bar(mp, x="Missed_Pmnt_Band", y="High_Risk_%",
                         color="High_Risk_%", color_continuous_scale="RdYlGn_r",
                         text="High_Risk_%",
                         title="High Risk Rate by Missed Payments",
                         labels={"Missed_Pmnt_Band":"Missed Payments"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # Income vs credit score scatter
        with col4:
            sample = df.sample(1500, random_state=42)
            fig = px.scatter(sample, x="Credit_Score", y="NETMONTHLYINCOME",
                             color="Risk_Category",
                             color_discrete_map={"Low Risk":"#1a6e3c",
                                                 "Medium Risk":"#2196F3",
                                                 "High Risk":"#e53935"},
                             opacity=0.5, title="Income vs Credit Score by Risk",
                             labels={"NETMONTHLYINCOME":"Income"})
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # Risk driver comparison table
        st.divider()
        st.markdown("**Risk Driver Summary — High Risk vs Low Risk**")
        summary = df.groupby("Is_High_Risk")[
            ["Credit_Score","NETMONTHLYINCOME","num_times_delinquent",
             "num_times_30p_dpd","Tot_Missed_Pmnt"]
        ].mean().round(2).reset_index()
        summary["Is_High_Risk"] = summary["Is_High_Risk"].map({0:"Low Risk",1:"High Risk"})
        summary.columns = ["Risk Group","Avg Credit Score","Avg Income",
                           "Avg Delinquencies","Avg 30+DPD","Avg Missed Pmnt"]
        st.dataframe(summary, use_container_width=True)

        # Customer summary table with filters
        st.divider()
        st.markdown("**Customer Loan Risk Summary Table**")
        f1, f2, f3 = st.columns(3)
        gender_filter = f1.multiselect("Filter by Gender",
                                        df["GENDER"].unique(),
                                        default=list(df["GENDER"].unique()))
        risk_filter   = f2.multiselect("Filter by Risk Level",
                                        df["Risk_Category"].unique(),
                                        default=list(df["Risk_Category"].unique()))
        age_filter    = f3.multiselect("Filter by Age Group",
                                        list(df["Age_Group"].dropna().unique()),
                                        default=list(df["Age_Group"].dropna().unique()))

        mask     = (df["GENDER"].isin(gender_filter) &
                    df["Risk_Category"].isin(risk_filter) &
                    df["Age_Group"].isin(age_filter))
        filtered = df[mask][["PROSPECTID","Credit_Score","NETMONTHLYINCOME",
                              "AGE","Risk_Category","Approved_Flag"]]
        filtered.columns = ["PROSPECTID","Credit Score","Monthly Income",
                             "Age","Risk Category","Approved Flag"]
        st.dataframe(filtered.head(100), use_container_width=True)
        st.caption(f"Showing top 100 of {len(filtered):,} filtered records")

    # ── PAGE 4: PREDICTIVE MODEL ──────────────────────────────
    elif page == "🤖 Predictive Model":
        st.title("🤖 Predictive Model — Logistic Regression")
        st.caption("Predicts: Is_High_Risk  |  P3/P4 = 1  |  P1/P2 = 0")
        st.divider()

        model, scaler, y_test, y_pred, y_prob, importance = train_model(df)

        report = classification_report(y_test, y_pred, output_dict=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",              f"{accuracy_score(y_test,y_pred)*100:.2f}%")
        c2.metric("ROC-AUC",               f"{roc_auc_score(y_test,y_prob):.4f}")
        c3.metric("Precision (High Risk)", f"{report['1']['precision']*100:.2f}%")
        c4.metric("Recall (High Risk)",    f"{report['1']['recall']*100:.2f}%")

        st.divider()
        col1, col2 = st.columns(2)

        # ROC Curve
        with col1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                     name=f"AUC = {auc:.3f}",
                                     line=dict(color="#2196F3", width=2)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                     line=dict(color="gray", dash="dash")))
            fig.update_layout(title="ROC Curve",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrix
        with col2:
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True,
                            x=["Pred: Low Risk","Pred: High Risk"],
                            y=["Act: Low Risk","Act: High Risk"],
                            color_continuous_scale="Blues",
                            title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        fig = px.bar(importance.sort_values("Importance"),
                     x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues",
                     title="Top 15 Feature Importances")
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # ── PAGE 5: RISK SCORER ───────────────────────────────────
    elif page == "🎯 Risk Scorer":
        st.title("🎯 Live Customer Risk Scorer")
        st.caption("Enter customer details to get an instant risk prediction")
        st.divider()

        model, scaler, *_ = train_model(df)

        with st.form("scorer"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Personal Info**")
                age          = st.slider("Age", 21, 80, 32)
                income       = st.number_input("Monthly Income (₹)", 0, 2500000, 25000, 1000)
                emp_tenure   = st.slider("Employment Tenure (months)", 0, 500, 60)
                credit_score = st.slider("Credit Score", 300, 900, 680)

            with c2:
                st.markdown("**Delinquency & Payments**")
                times_delinq = st.slider("Times Delinquent", 0, 50, 0)
                dpd_30       = st.slider("Times 30+ DPD",    0, 60, 0)
                dpd_60       = st.slider("Times 60+ DPD",    0, 60, 0)
                missed_pmnt  = st.slider("Missed Payments",  0, 100, 0)
                deliq_6m     = st.slider("Delinquencies L6M",0, 20, 0)
                deliq_12m    = st.slider("Delinquencies L12M",0,20, 0)

            with c3:
                st.markdown("**Account & Enquiries**")
                num_sub  = st.slider("Sub-Standard Accounts",0, 30, 0)
                num_dbt  = st.slider("Doubtful Accounts",    0, 20, 0)
                num_lss  = st.slider("Loss Accounts",        0, 10, 0)
                num_std  = st.slider("Standard Accounts",    0, 50, 5)
                tot_enq  = st.slider("Total Enquiries",      0, 100, 5)
                enq_l6m  = st.slider("Enquiries L6M",        0, 30, 1)
                enq_l3m  = st.slider("Enquiries L3M",        0, 20, 0)

            c4, c5 = st.columns(2)
            with c4:
                st.markdown("**Loan Flags**")
                cc_flag = st.selectbox("Credit Card?",  [0,1])
                pl_flag = st.selectbox("Personal Loan?",[0,1])
                hl_flag = st.selectbox("Home Loan?",    [0,1])
                gl_flag = st.selectbox("Gold Loan?",    [0,1])
                cc_util = st.slider("CC Utilisation %", 0.0, 100.0, 30.0)
                pl_util = st.slider("PL Utilisation %", 0.0, 100.0, 30.0)

            with c5:
                st.markdown("**Trade Lines**")
                total_tl      = st.slider("Total Trade Lines",  0, 100, 10)
                active_tl     = st.slider("Active Trade Lines", 0, 100, 5)
                age_oldest_tl = st.slider("Oldest TL (months)", 0, 400, 36)
                age_newest_tl = st.slider("Newest TL (months)", 0, 200, 6)

            predict = st.form_submit_button("🔍 Predict Risk", type="primary")

        if predict:
            values = [
                times_delinq, dpd_30, dpd_60, deliq_6m, deliq_12m,
                num_sub, num_dbt, num_lss, num_std,
                tot_enq, enq_l6m, enq_l3m,
                age, income, emp_tenure,
                cc_flag, pl_flag, hl_flag, gl_flag,
                total_tl, active_tl, missed_pmnt,
                age_oldest_tl, age_newest_tl,
                cc_util, pl_util, credit_score,
            ]
            inp  = pd.DataFrame([values], columns=FEATURES)
            prob = model.predict_proba(scaler.transform(inp))[0][1]
            pred = model.predict(scaler.transform(inp))[0]

            st.divider()
            if pred == 1:
                tier = "P3" if prob < 0.70 else "P4"
                st.error(f"🚨 HIGH RISK | Tier: **{tier}** | "
                         f"Probability: **{prob*100:.1f}%**\n\n"
                         "Recommendation: Manual review required.")
            else:
                tier = "P1" if prob < 0.20 else "P2"
                st.success(f"✅ LOW RISK | Tier: **{tier}** | "
                           f"Probability: **{prob*100:.1f}%**\n\n"
                           "Recommendation: Eligible for standard processing.")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob*100, 1),
                title={"text":"High Risk Probability (%)"},
                gauge={
                    "axis":  {"range":[0,100]},
                    "bar":   {"color":"#e53935" if pred==1 else "#1a6e3c"},
                    "steps": [
                        {"range":[0, 30],"color":"#c8e6c9"},
                        {"range":[30,60],"color":"#fff9c4"},
                        {"range":[60,100],"color":"#ffcdd2"},
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
