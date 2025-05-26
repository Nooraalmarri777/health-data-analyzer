
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Advanced Health Data Analyzer", layout="wide")
st.title("Advanced Health Data Analyzer")

uploaded_file = st.file_uploader(
    "Upload your health data file (CSV, Excel, TSV, JSON)", 
    type=["csv", "xlsx", "xls", "tsv", "json"]
)

def plot_categorical(df, col):
    counts = df[col].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_title(f"Bar chart of {col}")
    st.pyplot(fig)

def plot_numerical(df, col):
    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    df[col].hist(ax=axs[0], bins=20)
    axs[0].set_title(f"Histogram of {col}")
    sns.boxplot(x=df[col], ax=axs[1])
    axs[1].set_title(f"Boxplot of {col}")
    st.pyplot(fig)

def descriptive_stats(df):
    desc = pd.DataFrame()
    desc['Mean'] = df.mean()
    desc['Median'] = df.median()
    desc['Std Dev'] = df.std()
    desc['Range'] = df.max() - df.min()
    desc['Mean Abs Dev'] = df.mad()
    desc['25% Quantile'] = df.quantile(0.25)
    desc['75% Quantile'] = df.quantile(0.75)
    desc['IQR'] = desc['75% Quantile'] - desc['25% Quantile']
    desc['Coeff of Var (%)'] = (desc['Std Dev'] / desc['Mean']) * 100
    desc['Skewness'] = df.skew()
    desc['Kurtosis'] = df.kurtosis()
    return desc

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

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Automatic Data Analysis")

        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if date_cols and numeric_cols:
            st.markdown("### Time Series Analysis")
            for date_col in date_cols:
                df_sorted = df.sort_values(by=date_col)
                df_sorted = df_sorted.dropna(subset=[date_col])
                df_sorted.set_index(date_col, inplace=True)
                freq = st.selectbox(f"Select frequency for resampling based on '{date_col}'", ['D', 'W', 'M', 'Q', 'Y'], index=2)
                resampled = df_sorted[numeric_cols].resample(freq).mean()
                for col in numeric_cols:
                    st.line_chart(resampled[col], use_container_width=True)

        if len(numeric_cols) > 1:
            st.markdown("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if numeric_cols:
            st.markdown("### Numerical Data Distribution")
            for col in numeric_cols:
                plot_numerical(df, col)

            st.markdown("### Descriptive Statistics")
            stats = descriptive_stats(df[numeric_cols])
            st.dataframe(stats)

        if categorical_cols:
            st.markdown("### Categorical Data Distribution")
            for col in categorical_cols:
                plot_categorical(df, col)

        st.markdown("### Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.write("No missing values found.")
        else:
            st.dataframe(missing)

    except Exception as e:
        st.error(f"Error processing file: {e}")