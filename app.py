
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Health Data Analyzer", layout="wide")
st.title("Health Data Analyzer")

uploaded_file = st.file_uploader(
    "Upload your health data file (CSV, Excel, TSV, JSON)", 
    type=["csv", "xlsx", "xls", "tsv", "json"]
)

def show_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.write("No missing values detected.")
    else:
        st.write("Missing values per column:")
        st.dataframe(missing)

def show_top_values(df):
    st.write("Top 5 frequent values per column:")
    for col in df.columns:
        st.write(f"**{col}**")
        st.dataframe(df[col].value_counts().head())

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

        st.success(f"Successfully loaded {file_type.upper()} file.")
        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Report Settings")
        report_freq = st.selectbox("Select report frequency", ["Weekly", "Monthly", "Quarterly", "Yearly"])

        analysis_type = st.multiselect(
            "Select type(s) of analysis",
            [
                "Descriptive Statistics", 
                "Time Series Trends", 
                "Correlation Heatmap",
                "Histograms",
                "Missing Values Analysis",
                "Top Values Frequency",
                "Data Types Info",
                "Boxplots"
            ]
        )

        if st.button("Run Analysis"):
            st.subheader("Analysis Results")

            if "Descriptive Statistics" in analysis_type:
                st.markdown("### Descriptive Statistics")
                st.dataframe(df.describe())

            if "Time Series Trends" in analysis_type:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=['date'])
                    df.set_index('date', inplace=True)
                    freq = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}[report_freq]
                    numeric_cols = df.select_dtypes(include='number').columns
                    resampled_df = df[numeric_cols].resample(freq).mean()

                    st.markdown(f"### Time Series Trends ({report_freq})")
                    for col in numeric_cols:
                        st.line_chart(resampled_df[col], use_container_width=True)
                else:
                    st.warning("Time Series analysis requires a 'date' column.")

            if "Correlation Heatmap" in analysis_type:
                st.markdown("### Correlation Heatmap")
                numeric_cols = df.select_dtypes(include='number')
                if numeric_cols.shape[1] > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("Not enough numeric columns for correlation heatmap.")

            if "Histograms" in analysis_type:
                st.markdown("### Histograms")
                numeric_cols = df.select_dtypes(include='number').columns
                for col in numeric_cols:
                    fig, ax = plt.subplots()
                    df[col].hist(ax=ax, bins=20)
                    ax.set_title(f"Histogram of {col}")
                    st.pyplot(fig)

            if "Missing Values Analysis" in analysis_type:
                st.markdown("### Missing Values Analysis")
                show_missing_values(df)

            if "Top Values Frequency" in analysis_type:
                st.markdown("### Top Values Frequency")
                show_top_values(df)

            if "Data Types Info" in analysis_type:
                st.markdown("### Data Types Info")
                dtypes = pd.DataFrame(df.dtypes, columns=["Type"])
                st.dataframe(dtypes)

            if "Boxplots" in analysis_type:
                st.markdown("### Boxplots")
                numeric_cols = df.select_dtypes(include='number').columns
                for col in numeric_cols:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col], ax=ax)
                    ax.set_title(f"Boxplot of {col}")
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")