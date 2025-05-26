import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Health Data Analyzer", layout="wide")

st.title("Health Data Analyzer")

# Sidebar - Report options
st.sidebar.header("Report Options")
report_type = st.sidebar.selectbox("Select Report Type", ["Weekly", "Monthly", "Quarterly", "Yearly"])
analysis_type = st.sidebar.multiselect("Select Type of Analysis", ["Summary", "Statistical Measures", "Trends", "Gaps", "KPIs"])
chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar", "Line", "Box", "Histogram", "Pie"])

# File uploader
uploaded_file = st.file_uploader("Upload your health data file (CSV or Excel)", type=["csv", "xlsx"])

# Main logic
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Raw Data")
        st.dataframe(df)

        columns = st.multiselect("Select columns for analysis", df.columns.tolist(), default=df.columns.tolist())

        if "Summary" in analysis_type:
            st.subheader("Summary Statistics")
            st.write(df[columns].describe())

        if "Statistical Measures" in analysis_type:
            st.subheader("Statistical Measures")
            stats_df = pd.DataFrame({
                "Mean": df[columns].mean(),
                "Median": df[columns].median(),
                "Std Deviation": df[columns].std(),
                "Variance": df[columns].var(),
                "Min": df[columns].min(),
                "Max": df[columns].max(),
                "Range": df[columns].max() - df[columns].min(),
                "IQR": df[columns].quantile(0.75) - df[columns].quantile(0.25)
            })
            st.dataframe(stats_df)

        if "Trends" in analysis_type:
            st.subheader("Trend Visualization")
            time_col = st.selectbox("Select time-related column", df.columns)
            value_col = st.selectbox("Select value column", columns)
            trend_df = df[[time_col, value_col]].dropna()
            trend_df = trend_df.groupby(time_col)[value_col].mean().reset_index()

            if chart_type == "Line":
                fig, ax = plt.subplots()
                ax.plot(trend_df[time_col], trend_df[value_col])
                ax.set_title(f"{value_col} Over Time")
                st.pyplot(fig)

        if "KPIs" in analysis_type:
            st.subheader("Key Performance Indicators")
            kpi_col = st.selectbox("Select column for KPI visualization", columns)
            st.metric(label=f"Mean of {kpi_col}", value=round(df[kpi_col].mean(), 2))
            st.metric(label=f"Max of {kpi_col}", value=round(df[kpi_col].max(), 2))
            st.metric(label=f"Min of {kpi_col}", value=round(df[kpi_col].min(), 2))

        if "Gaps" in analysis_type:
            st.subheader("Missing Values and Gaps")
            st.write(df[columns].isnull().sum())

        # Chart customization
        st.subheader("Custom Chart")
        x_col = st.selectbox("X-axis", columns)
        y_col = st.selectbox("Y-axis", columns)
        title = st.text_input("Chart Title", "Custom Chart")
        color = st.color_picker("Chart Color", "#69b3a2")

        if chart_type == "Bar":
            fig, ax = plt.subplots()
            ax.bar(df[x_col], df[y_col], color=color)
            ax.set_title(title)
            st.pyplot(fig)
        elif chart_type == "Line":
            fig, ax = plt.subplots()
            ax.plot(df[x_col], df[y_col], color=color)
            ax.set_title(title)
            st.pyplot(fig)
        elif chart_type == "Box":
            fig, ax = plt.subplots()
            sns.boxplot(x=df[x_col], y=df[y_col], color=color, ax=ax)
            ax.set_title(title)
            st.pyplot(fig)
        elif chart_type == "Histogram":
            fig, ax = plt.subplots()
            ax.hist(df[y_col], bins=20, color=color)
            ax.set_title(title)
            st.pyplot(fig)
        elif chart_type == "Pie":
            pie_data = df[y_col].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, colors=[color]*len(pie_data), autopct="%1.1f%%")
            ax.set_title(title)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV or Excel file to get started.")