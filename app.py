
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Customizable Health Data Analyzer", layout="wide")
st.title("Customizable Health Data Analyzer")

uploaded_file = st.file_uploader(
    "Upload your health data file (CSV, Excel, TSV, JSON)", 
    type=["csv", "xlsx", "xls", "tsv", "json"]
)

def plot_histogram(df, col, title, color):
    fig, ax = plt.subplots()
    df[col].hist(ax=ax, bins=20, color=color)
    ax.set_title(title)
    st.pyplot(fig)

def plot_boxplot(df, col, title, color):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax, color=color)
    ax.set_title(title)
    st.pyplot(fig)

def plot_line(df, col, title, color):
    fig, ax = plt.subplots()
    df[col].plot.line(ax=ax, color=color)
    ax.set_title(title)
    st.pyplot(fig)

def plot_bar(df, col, title, color):
    fig, ax = plt.subplots()
    counts = df[col].value_counts()
    counts.plot(kind='bar', ax=ax, color=color)
    ax.set_title(title)
    st.pyplot(fig)

def plot_heatmap(df, cols, title, cmap):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[cols].corr(), annot=True, cmap=cmap, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

if uploaded_file:
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_type == 'tsv':
            df = pd.read_csv(uploaded_file, sep='\t')
        elif file_type == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        st.success(f"Loaded {file_type.upper()} file successfully!")
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # تحويل أي عمود يحتمل أن يكون تاريخ
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')

        st.subheader("Select Analyses to Perform")
        analyses = st.multiselect(
            "Choose analysis types",
            options=["Descriptive Statistics", "Time Series Analysis", "Correlation Heatmap", "Numerical Data Distribution", "Categorical Data Distribution"],
            default=["Descriptive Statistics"]
        )

        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Descriptive Statistics
        def descriptive_stats(df):
            desc = pd.DataFrame()
            desc['Mean'] = df.mean()
            desc['Median'] = df.median()
            desc['Std Dev'] = df.std()
            desc['Range'] = df.max() - df.min()
            desc['Mean Abs Dev'] = df.apply(lambda x: x.mad())
            desc['25% Quantile'] = df.quantile(0.25)
            desc['75% Quantile'] = df.quantile(0.75)
            desc['IQR'] = desc['75% Quantile'] - desc['25% Quantile']
            desc['Coeff of Var (%)'] = (desc['Std Dev'] / desc['Mean']) * 100
            desc['Skewness'] = df.skew()
            desc['Kurtosis'] = df.kurtosis()
            return desc

        if "Descriptive Statistics" in analyses and numeric_cols:
            st.subheader("Descriptive Statistics")
            stats = descriptive_stats(df[numeric_cols])
            st.dataframe(stats)

        if "Time Series Analysis" in analyses and date_cols and numeric_cols:
            st.subheader("Time Series Analysis")
            for date_col in date_cols:
                df_sorted = df.sort_values(by=date_col)
                df_sorted = df_sorted.dropna(subset=[date_col])
                df_sorted.set_index(date_col, inplace=True)
                freq = st.selectbox(f"Select frequency for resampling based on '{date_col}'", ['D', 'W', 'M', 'Q', 'Y'], index=2)
                resampled = df_sorted[numeric_cols].resample(freq).mean()
                for col in numeric_cols:
                    st.markdown(f"**{col}**")
                    st.line_chart(resampled[col], use_container_width=True)

        if "Correlation Heatmap" in analyses and len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            cmap = st.color_picker("Pick colormap for heatmap", "#FF5733")
            plot_heatmap(df, numeric_cols, "Correlation Heatmap", cmap)

        if "Numerical Data Distribution" in analyses and numeric_cols:
            st.subheader("Numerical Data Distribution")
            selected_num_cols = st.multiselect("Select numeric columns to plot", numeric_cols, default=numeric_cols[:2])
            chart_type_num = st.selectbox("Select chart type for numeric data", ["Histogram", "Boxplot"])
            color_num = st.color_picker("Pick color for numeric charts", "#1f77b4")
            for col in selected_num_cols:
                title = st.text_input(f"Chart title for {col}", f"{chart_type_num} of {col}")
                if chart_type_num == "Histogram":
                    plot_histogram(df, col, title, color_num)
                else:
                    plot_boxplot(df, col, title, color_num)

        if "Categorical Data Distribution" in analyses and categorical_cols:
            st.subheader("Categorical Data Distribution")
            selected_cat_cols = st.multiselect("Select categorical columns to plot", categorical_cols, default=categorical_cols[:2])
            color_cat = st.color_picker("Pick color for categorical charts", "#ff7f0e")
            for col in selected_cat_cols:
                title = st.text_input(f"Chart title for {col}", f"Bar chart of {col}")
                plot_bar(df, col, title, color_cat)

        st.markdown("### Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.write("No missing values found.")
        else:
            st.dataframe(missing)

    except Exception as e:
        st.error(f"Error processing file: {e}")