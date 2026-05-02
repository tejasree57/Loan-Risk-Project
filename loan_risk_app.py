# ============================================================
# LOAN RISK ANALYTICS — PYTHON ANALYSIS + STREAMLIT WEB APP
# Project  : Data-Powered Loan Risk Analytics
# Tech     : Python | Pandas | Scikit-learn | Plotly | Streamlit
# Sections :
#   1. Data Loading & Preprocessing
#   2. Descriptive Analytics
#   3. Diagnostic Analytics
#   4. Predictive Model (Logistic Regression)
#   5. Streamlit Web App (all sections as pages)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 : DATA LOADING & PREPROCESSING
# ============================================================

# -----------------------------------------------------------
# 1.1 Load data — reads from local Excel files.
#     In production, replace with SQL Server connection:
#     import pyodbc
#     conn = pyodbc.connect("DRIVER={SQL Server};SERVER=...;DATABASE=LoanRiskDB;")
#     cibil = pd.read_sql("SELECT * FROM vw_CIBIL_Cleaned", conn)
#     bank  = pd.read_sql("SELECT * FROM vw_Bank_Cleaned",  conn)
# -----------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    cibil = pd.read_excel("EXTERNAL_CIBIL_DATA.xlsx")
    bank  = pd.read_excel("INTERNAL_BANK_DATA.xlsx")
    df    = cibil.merge(bank, on="PROSPECTID", how="inner")
    return df

@st.cache_data(show_spinner="Preprocessing...")
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ----------------------------------------------------------
    # 1.2 Replace -99999 sentinel values with NaN
    #     These encode "not applicable" (e.g., never delinquent)
    # ----------------------------------------------------------
    sentinel_cols = [
        "time_since_recent_payment", "time_since_first_deliquency",
        "time_since_recent_deliquency", "max_delinquency_level",
        "max_deliq_6mts", "max_deliq_12mts",
        "tot_enq", "CC_enq", "CC_enq_L6m", "CC_enq_L12m",
        "PL_enq", "PL_enq_L6m", "PL_enq_L12m",
        "time_since_recent_enq", "enq_L12m", "enq_L6m", "enq_L3m",
        "CC_utilization", "PL_utilization", "max_unsec_exposure_inPct",
        "pct_currentBal_all_TL", "Age_Oldest_TL", "Age_Newest_TL",
    ]
    for col in sentinel_cols:
        if col in df.columns:
            df[col] = df[col].replace(-99999, np.nan)

    # ----------------------------------------------------------
    # 1.3 Fix outliers:
    #     pct_currentBal_all_TL > 100 (6 records) → cap at 100
    #     NETMONTHLYINCOME = 0 (46 records) → NaN
    # ----------------------------------------------------------
    if "pct_currentBal_all_TL" in df.columns:
        df["pct_currentBal_all_TL"] = df["pct_currentBal_all_TL"].clip(upper=100)

    df["NETMONTHLYINCOME"] = df["NETMONTHLYINCOME"].replace(0, np.nan)

    # ----------------------------------------------------------
    # 1.4 Derived / Feature-engineered columns
    # ----------------------------------------------------------
    # Binary risk target: P3 or P4 = High Risk (1), P1/P2 = Low Risk (0)
    df["Is_High_Risk"] = df["Approved_Flag"].apply(
        lambda x: 1 if x in ["P3", "P4"] else 0
    )

    # Credit Score band
    df["Credit_Score_Band"] = pd.cut(
        df["Credit_Score"],
        bins=[0, 599, 649, 699, 749, 900],
        labels=["Very Poor (<600)", "Poor (600-649)", "Fair (650-699)",
                "Good (700-749)", "Excellent (750+)"]
    )

    # Age group
    df["Age_Group"] = pd.cut(
        df["AGE"],
        bins=[20, 30, 40, 50, 60, 100],
        labels=["21-30", "31-40", "41-50", "51-60", "60+"]
    )

    # Income band (INR monthly)
    df["Income_Band"] = pd.cut(
        df["NETMONTHLYINCOME"].fillna(-1),
        bins=[-2, 0, 15000, 30000, 60000, 1e8],
        labels=["Unknown/Zero", "Low (<15K)", "Mid (15K-30K)",
                "Upper-Mid (30K-60K)", "High (>60K)"]
    )

    # Missed payment band
    df["Missed_Pmnt_Band"] = pd.cut(
        df["Tot_Missed_Pmnt"],
        bins=[-1, 0, 2, 5, 1000],
        labels=["No Missed", "Low (1-2)", "Moderate (3-5)", "High (6+)"]
    )

    # Ever delinquent flag
    df["Ever_Delinquent"] = df["max_delinquency_level"].notna().astype(int)

    # Trade line activity ratio
    df["TL_Activity_Ratio"] = np.where(
        df["Total_TL"] > 0,
        df["Tot_Active_TL"] / df["Total_TL"],
        np.nan
    )

    return df


# ============================================================
# SECTION 2 : DESCRIPTIVE ANALYTICS FUNCTIONS
# ============================================================

def descriptive_overview(df):
    """Returns a dict of key summary stats for display."""
    total     = len(df)
    high_risk = df["Is_High_Risk"].sum()
    return {
        "Total Customers":      f"{total:,}",
        "High Risk (P3/P4)":    f"{high_risk:,}  ({high_risk/total*100:.1f}%)",
        "Low Risk (P1/P2)":     f"{total-high_risk:,}  ({(total-high_risk)/total*100:.1f}%)",
        "Avg Credit Score":     f"{df['Credit_Score'].mean():.1f}",
        "Avg Monthly Income":   f"₹{df['NETMONTHLYINCOME'].mean():,.0f}",
        "Avg Age":              f"{df['AGE'].mean():.1f} yrs",
        "Ever Delinquent":      f"{df['Ever_Delinquent'].sum():,}  ({df['Ever_Delinquent'].mean()*100:.1f}%)",
        "Avg Missed Payments":  f"{df['Tot_Missed_Pmnt'].mean():.2f}",
    }


def plot_approved_flag(df):
    counts = df["Approved_Flag"].value_counts().reset_index()
    counts.columns = ["Tier", "Count"]
    counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    fig = px.bar(
        counts, x="Tier", y="Count", text="Pct",
        color="Tier",
        color_discrete_map={"P1":"#1a6e3c","P2":"#2196F3","P3":"#FF9800","P4":"#e53935"},
        title="Approval Tier Distribution",
        labels={"Count":"Number of Customers", "Tier":"Approval Tier"}
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="white")
    return fig


def plot_credit_score_dist(df):
    fig = px.histogram(
        df, x="Credit_Score", nbins=50,
        color="Approved_Flag",
        color_discrete_map={"P1":"#1a6e3c","P2":"#2196F3","P3":"#FF9800","P4":"#e53935"},
        barmode="overlay", opacity=0.7,
        title="Credit Score Distribution by Approval Tier",
        labels={"Credit_Score":"Credit Score", "count":"Customers"}
    )
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_age_group(df):
    grp = (
        df.groupby("Age_Group", observed=True)
          .agg(Count=("PROSPECTID","count"),
               High_Risk_Rate=("Is_High_Risk","mean"))
          .reset_index()
    )
    grp["High_Risk_Rate"] *= 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=grp["Age_Group"].astype(str), y=grp["Count"],
        name="Customers", marker_color="#2196F3"), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=grp["Age_Group"].astype(str), y=grp["High_Risk_Rate"],
        name="High Risk %", mode="lines+markers",
        marker=dict(color="#e53935", size=8),
        line=dict(color="#e53935", width=2)), secondary_y=True)
    fig.update_layout(
        title="Age Group — Customer Count & High Risk Rate",
        plot_bgcolor="white"
    )
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="High Risk Rate (%)", secondary_y=True)
    return fig


def plot_income_distribution(df):
    fig = px.box(
        df, x="Approved_Flag", y="NETMONTHLYINCOME",
        color="Approved_Flag",
        color_discrete_map={"P1":"#1a6e3c","P2":"#2196F3","P3":"#FF9800","P4":"#e53935"},
        title="Monthly Income Distribution by Approval Tier",
        labels={"NETMONTHLYINCOME":"Net Monthly Income (₹)", "Approved_Flag":"Tier"}
    )
    fig.update_layout(plot_bgcolor="white", showlegend=False)
    return fig


def plot_education_risk(df):
    grp = (
        df.groupby("EDUCATION")
          .agg(Count=("PROSPECTID","count"),
               High_Risk_Rate=("Is_High_Risk","mean"),
               Avg_Score=("Credit_Score","mean"))
          .reset_index()
          .sort_values("High_Risk_Rate", ascending=False)
    )
    grp["High_Risk_Rate"] *= 100
    fig = px.bar(
        grp, x="EDUCATION", y="High_Risk_Rate",
        color="High_Risk_Rate",
        color_continuous_scale="RdYlGn_r",
        text="High_Risk_Rate",
        title="High Risk Rate by Education Level",
        labels={"High_Risk_Rate":"High Risk Rate (%)", "EDUCATION":"Education"}
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_product_penetration(df):
    products = {
        "Credit Card": df["CC_Flag"].sum(),
        "Personal Loan": df["PL_Flag"].sum(),
        "Home Loan": df["HL_Flag"].sum(),
        "Gold Loan": df["GL_Flag"].sum(),
    }
    total = len(df)
    pdata = pd.DataFrame({
        "Product": list(products.keys()),
        "Holders": list(products.values()),
        "Pct": [v/total*100 for v in products.values()]
    })
    fig = px.bar(
        pdata, x="Product", y="Pct",
        text="Pct", color="Product",
        title="Loan Product Penetration (%)",
        labels={"Pct":"% of Customers"}
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(plot_bgcolor="white", showlegend=False)
    return fig


def plot_delinquency_summary(df):
    deliq_grp = (
        df.groupby("Approved_Flag")
          .agg(
              Ever_Delinquent_Pct=("Ever_Delinquent","mean"),
              Avg_30p_DPD=("num_times_30p_dpd","mean"),
              Avg_60p_DPD=("num_times_60p_dpd","mean"),
          )
          .reset_index()
    )
    deliq_grp["Ever_Delinquent_Pct"] *= 100
    fig = px.bar(
        deliq_grp, x="Approved_Flag", y="Ever_Delinquent_Pct",
        color="Approved_Flag",
        color_discrete_map={"P1":"#1a6e3c","P2":"#2196F3","P3":"#FF9800","P4":"#e53935"},
        text="Ever_Delinquent_Pct",
        title="% Ever Delinquent by Approval Tier",
        labels={"Ever_Delinquent_Pct":"% Ever Delinquent", "Approved_Flag":"Tier"}
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(plot_bgcolor="white", showlegend=False)
    return fig


# ============================================================
# SECTION 3 : DIAGNOSTIC ANALYTICS FUNCTIONS
# ============================================================

def plot_risk_driver_comparison(df):
    """Bar chart comparing key metrics between High Risk & Low Risk."""
    drivers = {
        "Avg Delinquencies":     ("num_times_delinquent", 1),
        "Avg 30+ DPD":           ("num_times_30p_dpd", 1),
        "Avg 60+ DPD":           ("num_times_60p_dpd", 1),
        "Avg Missed Payments":   ("Tot_Missed_Pmnt", 1),
        "Avg Credit Score":      ("Credit_Score", 0.01),   # scaled for chart
        "Avg Income (÷1000)":    ("NETMONTHLYINCOME", 0.001),
    }
    rows = []
    for label, (col, scale) in drivers.items():
        hr = df[df["Is_High_Risk"]==1][col].mean() * scale
        lr = df[df["Is_High_Risk"]==0][col].mean() * scale
        rows.append({"Driver": label, "High Risk": round(hr,2), "Low Risk": round(lr,2)})
    comp = pd.DataFrame(rows)
    comp_melt = comp.melt(id_vars="Driver", var_name="Risk Group", value_name="Value")
    fig = px.bar(
        comp_melt, x="Driver", y="Value", color="Risk Group",
        barmode="group",
        color_discrete_map={"High Risk":"#e53935","Low Risk":"#1a6e3c"},
        title="Key Risk Drivers — High Risk vs Low Risk",
        labels={"Value":"Average Value (scaled)"}
    )
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_delinquency_vs_risk(df):
    grp = (
        df.groupby("num_times_delinquent")
          .agg(High_Risk_Rate=("Is_High_Risk","mean"), Count=("PROSPECTID","count"))
          .reset_index()
    )
    grp["High_Risk_Rate"] *= 100
    grp = grp[grp["Count"] >= 50]   # only stable groups
    fig = px.scatter(
        grp, x="num_times_delinquent", y="High_Risk_Rate",
        size="Count", color="High_Risk_Rate",
        color_continuous_scale="RdYlGn_r",
        title="Delinquency Count vs High Risk Rate",
        labels={"num_times_delinquent":"Times Delinquent",
                "High_Risk_Rate":"High Risk Rate (%)"}
    )
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_credit_score_vs_missed_pmnt(df):
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Credit_Score", y="Tot_Missed_Pmnt",
        color="Approved_Flag",
        color_discrete_map={"P1":"#1a6e3c","P2":"#2196F3","P3":"#FF9800","P4":"#e53935"},
        opacity=0.5,
        title="Credit Score vs Missed Payments (sample 3,000)",
        labels={"Tot_Missed_Pmnt":"Total Missed Payments",
                "Credit_Score":"Credit Score"}
    )
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_income_vs_risk(df):
    grp = (
        df.groupby("Income_Band", observed=True)
          .agg(High_Risk_Rate=("Is_High_Risk","mean"),
               Count=("PROSPECTID","count"))
          .reset_index()
    )
    grp["High_Risk_Rate"] *= 100
    fig = px.bar(
        grp, x="Income_Band", y="High_Risk_Rate",
        color="High_Risk_Rate", text="High_Risk_Rate",
        color_continuous_scale="RdYlGn_r",
        title="High Risk Rate by Income Band",
        labels={"High_Risk_Rate":"High Risk Rate (%)", "Income_Band":"Income Band"}
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_missed_pmnt_vs_risk(df):
    grp = (
        df.groupby("Missed_Pmnt_Band", observed=True)
          .agg(High_Risk_Rate=("Is_High_Risk","mean"),
               Count=("PROSPECTID","count"))
          .reset_index()
    )
    grp["High_Risk_Rate"] *= 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=grp["Missed_Pmnt_Band"].astype(str), y=grp["Count"],
        name="Customers", marker_color="#2196F3"), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=grp["Missed_Pmnt_Band"].astype(str), y=grp["High_Risk_Rate"],
        name="High Risk %", mode="lines+markers",
        marker=dict(color="#e53935", size=9),
        line=dict(color="#e53935", width=2)), secondary_y=True)
    fig.update_layout(
        title="Missed Payments Band — Customers & High Risk Rate",
        plot_bgcolor="white"
    )
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="High Risk Rate (%)", secondary_y=True)
    return fig


def plot_substandard_accounts(df):
    grp = (
        df.groupby("Approved_Flag")
          .agg(
              Avg_Sub=("num_sub","mean"),
              Avg_Dbt=("num_dbt","mean"),
              Avg_Lss=("num_lss","mean"),
              Avg_Std=("num_std","mean"),
          )
          .reset_index()
    )
    melt = grp.melt(
        id_vars="Approved_Flag",
        value_vars=["Avg_Sub","Avg_Dbt","Avg_Lss","Avg_Std"],
        var_name="Account_Type", value_name="Avg_Count"
    )
    melt["Account_Type"] = melt["Account_Type"].map({
        "Avg_Sub":"Sub-Standard","Avg_Dbt":"Doubtful",
        "Avg_Lss":"Loss","Avg_Std":"Standard"
    })
    fig = px.bar(
        melt, x="Approved_Flag", y="Avg_Count",
        color="Account_Type", barmode="group",
        title="Avg Credit Classification Accounts by Approval Tier",
        labels={"Avg_Count":"Avg Number of Accounts","Approved_Flag":"Tier"}
    )
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_enquiry_surge(df):
    grp = (
        df.dropna(subset=["enq_L3m","enq_L12m"])
          .assign(surge=lambda x: np.where(
              x["enq_L12m"]>0, x["enq_L3m"]/x["enq_L12m"], 0))
          .groupby("Approved_Flag")
          .agg(Avg_Surge=("surge","mean"))
          .reset_index()
    )
    fig = px.bar(
        grp, x="Approved_Flag", y="Avg_Surge",
        color="Approved_Flag",
        color_discrete_map={"P1":"#1a6e3c","P2":"#2196F3","P3":"#FF9800","P4":"#e53935"},
        text="Avg_Surge",
        title="Enquiry Surge Ratio (L3m / L12m) by Approval Tier",
        labels={"Avg_Surge":"Avg Enquiry Surge Ratio","Approved_Flag":"Tier"}
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(plot_bgcolor="white", showlegend=False)
    return fig


# ============================================================
# SECTION 4 : PREDICTIVE MODEL — LOGISTIC REGRESSION
# ============================================================

FEATURE_COLS = [
    # Delinquency
    "num_times_delinquent", "num_times_30p_dpd", "num_times_60p_dpd",
    "num_deliq_6mts", "num_deliq_12mts",
    # Credit classification
    "num_sub", "num_dbt", "num_lss", "num_std",
    # Enquiries
    "tot_enq", "enq_L6m", "enq_L3m",
    # Demographics
    "AGE", "NETMONTHLYINCOME", "Time_With_Curr_Empr",
    # Loan flags
    "CC_Flag", "PL_Flag", "HL_Flag", "GL_Flag",
    # Trade lines (bank)
    "Total_TL", "Tot_Active_TL", "Tot_Missed_Pmnt",
    "Age_Oldest_TL", "Age_Newest_TL",
    # Utilization
    "CC_utilization", "PL_utilization",
    # Credit score
    "Credit_Score",
]
TARGET_COL = "Is_High_Risk"


@st.cache_resource(show_spinner="Training model...")
def train_model(df: pd.DataFrame):
    """
    Train a Logistic Regression classifier to predict Is_High_Risk.
    Returns: model, scaler, X_test, y_test, y_pred, y_prob, feature_importances
    """
    # Select features and drop rows with NaN in key columns
    feat_available = [c for c in FEATURE_COLS if c in df.columns]
    data = df[feat_available + [TARGET_COL]].dropna()

    X = data[feat_available]
    y = data[TARGET_COL]

    # Train/test split — 80/20, stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardise features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Logistic Regression with class weight balancing
    model = LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight="balanced", C=0.1
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # Feature importance (absolute coefficients)
    importance = pd.DataFrame({
        "Feature":    feat_available,
        "Importance": np.abs(model.coef_[0])
    }).sort_values("Importance", ascending=False).head(15)

    return model, scaler, feat_available, X_test, y_test, y_pred, y_prob, importance


def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC Curve (AUC = {auc:.3f})",
        line=dict(color="#2196F3", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        name="Random Classifier",
        line=dict(color="gray", dash="dash")
    ))
    fig.update_layout(
        title=f"ROC Curve — AUC = {auc:.3f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor="white"
    )
    return fig, auc


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Low Risk (0)", "High Risk (1)"]
    fig = px.imshow(
        cm, text_auto=True, aspect="auto",
        x=labels, y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return fig


def plot_feature_importance(importance):
    fig = px.bar(
        importance.sort_values("Importance"), x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Blues",
        title="Top 15 Feature Importances (|Coefficient|)"
    )
    fig.update_layout(plot_bgcolor="white")
    return fig


# ============================================================
# SECTION 5 : STREAMLIT WEB APPLICATION
# ============================================================

def main():
    # -----------------------------------------------------------
    # App config & styling
    # -----------------------------------------------------------
    st.set_page_config(
        page_title="Loan Risk Analytics",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS — brand colours matching the project deck
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #0d2137 60%, #007b8e);
            color: white; padding: 20px 30px; border-radius: 8px;
            margin-bottom: 20px;
        }
        .main-header h1 { margin:0; font-size:2rem; }
        .main-header p  { margin:4px 0 0; font-size:1rem; opacity:0.85; }
        .metric-card {
            background: #f5f7fa; border-left: 5px solid #007b8e;
            padding: 16px 20px; border-radius: 6px; margin-bottom: 12px;
        }
        .metric-label { font-size:0.82rem; color:#555; font-weight:600; }
        .metric-value { font-size:1.5rem; font-weight:700; color:#0d2137; }
        .section-title {
            font-size:1.3rem; font-weight:700; color:#0d2137;
            border-bottom: 3px solid #f5a623; padding-bottom:6px;
            margin: 20px 0 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------
    # Sidebar navigation
    # -----------------------------------------------------------
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/bank-building.png", width=60)
        st.markdown("## 🏦 Loan Risk Analytics")
        st.markdown("---")
        page = st.selectbox(
            "Navigate to",
            ["🏠 Overview",
             "📊 Descriptive Analytics",
             "🔍 Diagnostic Analytics",
             "🤖 Predictive Model",
             "🎯 Risk Scorer"]
        )
        st.markdown("---")
        st.caption("Data: CIBIL + Internal Bank | 51,336 Records | 88 Features")

    # -----------------------------------------------------------
    # Load & preprocess data
    # -----------------------------------------------------------
    try:
        raw_df = load_data()
    except FileNotFoundError:
        st.error(
            "❌ Data files not found.\n\n"
            "Place `EXTERNAL_CIBIL_DATA.xlsx` and `INTERNAL_BANK_DATA.xlsx` "
            "in the same folder as this script, then rerun."
        )
        st.stop()

    df = preprocess(raw_df)

    # -----------------------------------------------------------
    # PAGE 1 : OVERVIEW
    # -----------------------------------------------------------
    if page == "🏠 Overview":
        st.markdown("""
        <div class="main-header">
            <h1>🏦 Loan Risk Analytics Dashboard</h1>
            <p>Data-Powered Lending Decisions | Indian Retail Banking | 2026</p>
        </div>
        """, unsafe_allow_html=True)

        stats = descriptive_overview(df)
        cols = st.columns(4)
        for i, (label, value) in enumerate(stats.items()):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Approval Tier Distribution</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_approved_flag(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_credit_score_dist(df), use_container_width=True)

        st.markdown('<div class="section-title">Risk Profile Snapshot</div>',
                    unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_product_penetration(df), use_container_width=True)
        with c4:
            st.plotly_chart(plot_delinquency_summary(df), use_container_width=True)

        st.info(
            "ℹ️ **P1** = Best (lowest risk) | **P2** = Moderate | "
            "**P3** = Elevated | **P4** = High Risk (worst). "
            "P3 + P4 are treated as **High Risk** in this analysis."
        )

    # -----------------------------------------------------------
    # PAGE 2 : DESCRIPTIVE ANALYTICS
    # -----------------------------------------------------------
    elif page == "📊 Descriptive Analytics":
        st.markdown("## 📊 Descriptive Analytics")
        st.caption("Understanding who our customers are — distributions, profiles, and patterns")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Demographics", "Income & Employment", "Credit Profile", "Trade Lines"]
        )

        with tab1:
            st.markdown('<div class="section-title">Age Group Distribution & Risk</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(plot_age_group(df), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                gender_grp = df.groupby("GENDER").agg(
                    Count=("PROSPECTID","count"),
                    High_Risk_Rate=("Is_High_Risk","mean"),
                    Avg_Score=("Credit_Score","mean")
                ).reset_index()
                gender_grp["High_Risk_Rate"] = (gender_grp["High_Risk_Rate"]*100).round(1)
                gender_grp["Avg_Score"]       = gender_grp["Avg_Score"].round(1)
                fig = px.pie(gender_grp, values="Count", names="GENDER",
                             title="Gender Split", hole=0.4,
                             color_discrete_sequence=["#2196F3","#FF9800"])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.plotly_chart(plot_education_risk(df), use_container_width=True)

            ms_grp = df.groupby("MARITALSTATUS").agg(
                Count=("PROSPECTID","count"),
                High_Risk_Pct=("Is_High_Risk","mean")
            ).reset_index()
            ms_grp["High_Risk_Pct"] = (ms_grp["High_Risk_Pct"]*100).round(2)
            st.dataframe(ms_grp, use_container_width=True)

        with tab2:
            st.plotly_chart(plot_income_distribution(df), use_container_width=True)
            st.plotly_chart(plot_income_vs_risk(df), use_container_width=True)
            emp_grp = df.groupby(
                pd.cut(df["Time_With_Curr_Empr"],
                       bins=[-1,12,36,60,120,10000],
                       labels=["<1yr","1-3yr","3-5yr","5-10yr","10+yr"])
            ).agg(
                Count=("PROSPECTID","count"),
                High_Risk_Pct=("Is_High_Risk","mean"),
                Avg_Income=("NETMONTHLYINCOME","mean")
            ).reset_index()
            emp_grp.columns = ["Tenure Band","Count","High Risk %","Avg Income"]
            emp_grp["High Risk %"] = (emp_grp["High Risk %"]*100).round(1)
            emp_grp["Avg Income"]  = emp_grp["Avg Income"].round(0)
            st.markdown("**Employment Tenure vs Risk**")
            st.dataframe(emp_grp, use_container_width=True)

        with tab3:
            st.plotly_chart(plot_credit_score_dist(df), use_container_width=True)
            score_grp = df.groupby("Credit_Score_Band", observed=True).agg(
                Count=("PROSPECTID","count"),
                High_Risk_Pct=("Is_High_Risk","mean"),
                Avg_Income=("NETMONTHLYINCOME","mean")
            ).reset_index()
            score_grp["High_Risk_Pct"] = (score_grp["High_Risk_Pct"]*100).round(1)
            score_grp["Avg_Income"]     = score_grp["Avg_Income"].round(0)
            st.dataframe(score_grp, use_container_width=True)

            st.plotly_chart(plot_delinquency_summary(df), use_container_width=True)

        with tab4:
            tl_stats = df[[
                "Total_TL","Tot_Active_TL","Tot_Closed_TL",
                "Tot_Missed_Pmnt","Age_Oldest_TL","Age_Newest_TL"
            ]].describe().T.reset_index()
            tl_stats.columns = ["Column","Count","Mean","Std","Min","25%","50%","75%","Max"]
            tl_stats = tl_stats.applymap(
                lambda x: round(x, 2) if isinstance(x, float) else x
            )
            st.markdown("**Trade Line Summary Statistics**")
            st.dataframe(tl_stats, use_container_width=True)

            st.plotly_chart(plot_missed_pmnt_vs_risk(df), use_container_width=True)

    # -----------------------------------------------------------
    # PAGE 3 : DIAGNOSTIC ANALYTICS
    # -----------------------------------------------------------
    elif page == "🔍 Diagnostic Analytics":
        st.markdown("## 🔍 Diagnostic Analytics")
        st.caption("Why do customers default? Root cause analysis and risk factor identification")

        st.plotly_chart(plot_risk_driver_comparison(df), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_delinquency_vs_risk(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_credit_score_vs_missed_pmnt(df), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_missed_pmnt_vs_risk(df), use_container_width=True)
        with c4:
            st.plotly_chart(plot_enquiry_surge(df), use_container_width=True)

        st.plotly_chart(plot_substandard_accounts(df), use_container_width=True)

        # Product flag vs risk table
        st.markdown('<div class="section-title">Loan Product vs Risk Rate</div>',
                    unsafe_allow_html=True)
        prod_rows = []
        for flag, name in [("CC_Flag","Credit Card"),("PL_Flag","Personal Loan"),
                            ("HL_Flag","Home Loan"),("GL_Flag","Gold Loan")]:
            sub = df[df[flag]==1]
            prod_rows.append({
                "Product": name,
                "Holders": len(sub),
                "High Risk Count": sub["Is_High_Risk"].sum(),
                "High Risk Rate (%)": round(sub["Is_High_Risk"].mean()*100, 2)
            })
        st.dataframe(pd.DataFrame(prod_rows), use_container_width=True)

        # Heatmap: Gender × Age Group risk
        st.markdown('<div class="section-title">Risk Heatmap — Gender × Age Group</div>',
                    unsafe_allow_html=True)
        hm = df.groupby(
            ["GENDER", "Age_Group"], observed=True
        )["Is_High_Risk"].mean().reset_index()
        hm["High_Risk_Pct"] = (hm["Is_High_Risk"] * 100).round(1)
        hm_pivot = hm.pivot(index="GENDER", columns="Age_Group", values="High_Risk_Pct")
        fig_hm = px.imshow(
            hm_pivot, text_auto=True, color_continuous_scale="RdYlGn_r",
            title="High Risk Rate (%) by Gender × Age Group",
            zmin=0, zmax=60
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # -----------------------------------------------------------
    # PAGE 4 : PREDICTIVE MODEL
    # -----------------------------------------------------------
    elif page == "🤖 Predictive Model":
        st.markdown("## 🤖 Predictive Model — Logistic Regression")
        st.caption("Binary classification: Is_High_Risk (P3/P4 = 1, P1/P2 = 0)")

        st.info(
            "**Model:** Logistic Regression | **Features:** 27 | "
            "**Split:** 80% Train / 20% Test | **Balancing:** class_weight='balanced'"
        )

        model, scaler, feat_cols, X_test, y_test, y_pred, y_prob, importance = \
            train_model(df)

        # Key metrics
        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{acc*100:.2f}%")
        m2.metric("ROC-AUC",   f"{auc:.4f}")
        m3.metric("Precision (High Risk)", f"{report_dict['1']['precision']*100:.2f}%")
        m4.metric("Recall (High Risk)",    f"{report_dict['1']['recall']*100:.2f}%")

        c1, c2 = st.columns(2)
        with c1:
            roc_fig, _ = plot_roc_curve(y_test, y_prob)
            st.plotly_chart(roc_fig, use_container_width=True)
        with c2:
            st.plotly_chart(plot_confusion_matrix(y_test, y_pred),
                            use_container_width=True)

        st.plotly_chart(plot_feature_importance(importance), use_container_width=True)

        # Full classification report
        st.markdown("**Classification Report**")
        rpt = pd.DataFrame(report_dict).T.reset_index()
        rpt.columns = ["Class","Precision","Recall","F1-Score","Support"]
        rpt = rpt[rpt["Class"].isin(["0","1","macro avg","weighted avg"])]
        rpt["Class"] = rpt["Class"].map(
            {"0":"Low Risk (P1/P2)","1":"High Risk (P3/P4)",
             "macro avg":"Macro Avg","weighted avg":"Weighted Avg"}
        )
        st.dataframe(rpt.round(4), use_container_width=True)

    # -----------------------------------------------------------
    # PAGE 5 : RISK SCORER (Live Prediction)
    # -----------------------------------------------------------
    elif page == "🎯 Risk Scorer":
        st.markdown("## 🎯 Live Customer Risk Scorer")
        st.caption(
            "Enter a customer's details below to get an instant high-risk prediction."
        )

        model, scaler, feat_cols, *_ = train_model(df)

        with st.form("risk_form"):
            st.markdown("### 👤 Customer Profile")
            c1, c2, c3 = st.columns(3)

            with c1:
                age             = st.slider("Age",              21, 80, 32)
                income          = st.number_input("Net Monthly Income (₹)", 0, 2500000, 25000, step=1000)
                emp_tenure      = st.slider("Employment Tenure (months)", 0, 500, 60)
                credit_score    = st.slider("Credit Score",     300, 900, 680)

            with c2:
                times_delinq    = st.slider("Times Delinquent (ever)",  0, 50, 0)
                dpd_30          = st.slider("Times 30+ DPD",            0, 60, 0)
                dpd_60          = st.slider("Times 60+ DPD",            0, 60, 0)
                missed_pmnt     = st.slider("Total Missed Payments",    0, 100, 0)
                deliq_6m        = st.slider("Delinquencies (last 6M)",  0, 20, 0)
                deliq_12m       = st.slider("Delinquencies (last 12M)", 0, 20, 0)

            with c3:
                num_sub         = st.slider("Sub-Standard Accounts",   0, 30, 0)
                num_dbt         = st.slider("Doubtful Accounts",        0, 20, 0)
                num_lss         = st.slider("Loss Accounts",            0, 10, 0)
                num_std         = st.slider("Standard Accounts",        0, 50, 5)
                tot_enq         = st.slider("Total Enquiries",          0, 100, 5)
                enq_l6m         = st.slider("Enquiries (L6M)",          0, 30, 1)
                enq_l3m         = st.slider("Enquiries (L3M)",          0, 20, 0)

            st.markdown("### 🏦 Loan & Trade Line Details")
            c4, c5 = st.columns(2)
            with c4:
                cc_flag         = st.selectbox("Credit Card Holder",  [0,1], index=0)
                pl_flag         = st.selectbox("Personal Loan Holder",[0,1], index=0)
                hl_flag         = st.selectbox("Home Loan Holder",    [0,1], index=0)
                gl_flag         = st.selectbox("Gold Loan Holder",    [0,1], index=0)
                cc_util         = st.slider("CC Utilisation (%)", 0.0, 100.0, 30.0)
                pl_util         = st.slider("PL Utilisation (%)", 0.0, 100.0, 30.0)
            with c5:
                total_tl        = st.slider("Total Trade Lines",        0, 100, 10)
                active_tl       = st.slider("Active Trade Lines",       0, 100, 5)
                age_oldest_tl   = st.slider("Age of Oldest TL (months)",0, 400, 36)
                age_newest_tl   = st.slider("Age of Newest TL (months)",0, 200, 6)

            submitted = st.form_submit_button("🔍 Predict Risk", type="primary")

        if submitted:
            # Build input vector matching FEATURE_COLS order
            input_map = {
                "num_times_delinquent": times_delinq,
                "num_times_30p_dpd":    dpd_30,
                "num_times_60p_dpd":    dpd_60,
                "num_deliq_6mts":       deliq_6m,
                "num_deliq_12mts":      deliq_12m,
                "num_sub":              num_sub,
                "num_dbt":              num_dbt,
                "num_lss":              num_lss,
                "num_std":              num_std,
                "tot_enq":              tot_enq,
                "enq_L6m":              enq_l6m,
                "enq_L3m":              enq_l3m,
                "AGE":                  age,
                "NETMONTHLYINCOME":     income,
                "Time_With_Curr_Empr":  emp_tenure,
                "CC_Flag":              cc_flag,
                "PL_Flag":              pl_flag,
                "HL_Flag":              hl_flag,
                "GL_Flag":              gl_flag,
                "Total_TL":             total_tl,
                "Tot_Active_TL":        active_tl,
                "Tot_Missed_Pmnt":      missed_pmnt,
                "Age_Oldest_TL":        age_oldest_tl,
                "Age_Newest_TL":        age_newest_tl,
                "CC_utilization":       cc_util,
                "PL_utilization":       pl_util,
                "Credit_Score":         credit_score,
            }
            input_row = pd.DataFrame([[input_map[c] for c in feat_cols]], columns=feat_cols)
            input_scaled = scaler.transform(input_row)
            prob   = model.predict_proba(input_scaled)[0][1]
            pred   = model.predict(input_scaled)[0]

            st.markdown("---")
            if pred == 1:
                risk_tier = "P3" if prob < 0.70 else "P4"
                st.error(
                    f"🚨 **HIGH RISK Customer** | Predicted Tier: **{risk_tier}**\n\n"
                    f"Default Probability: **{prob*100:.1f}%**\n\n"
                    "Recommendation: Manual review required before loan approval."
                )
            else:
                risk_tier = "P1" if prob < 0.20 else "P2"
                st.success(
                    f"✅ **LOW RISK Customer** | Predicted Tier: **{risk_tier}**\n\n"
                    f"Default Probability: **{prob*100:.1f}%**\n\n"
                    "Recommendation: Eligible for standard loan processing."
                )

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob*100, 1),
                title={"text": "High Risk Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#e53935" if pred==1 else "#1a6e3c"},
                    "steps": [
                        {"range": [0,  30], "color": "#c8e6c9"},
                        {"range": [30, 60], "color": "#fff9c4"},
                        {"range": [60,100], "color": "#ffcdd2"},
                    ],
                    "threshold": {
                        "line": {"color": "black","width": 3},
                        "value": 50
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)


if __name__ == "__main__":
    main()
