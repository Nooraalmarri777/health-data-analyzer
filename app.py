
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Health Data Analyzer", layout="wide")

st.title("Health Data Analyzer")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your health data file (CSV format)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Step 2: Choose Report Frequency
        st.subheader("Report Settings")
        report_freq = st.selectbox("Select report frequency", ["Weekly", "Monthly", "Quarterly", "Yearly"])

        # Step 3: Choose Analysis Type
        analysis_type = st.multiselect(
            "Select type of analysis",
            ["Descriptive Statistics", "Time Series Trends", "Correlation Heatmap"]
        )

        # Step 4: Trigger analysis
        if st.button("Run Analysis"):
            st.subheader("Analysis Results")

            if "Descriptive Statistics" in analysis_type:
                st.markdown("### Descriptive Statistics")
                st.dataframe(df.describe())

            if "Time Series Trends" in analysis_type:
                # Assuming there is a 'date' column
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    if report_freq == "Monthly":
                        freq = "M"
                    elif report_freq == "Weekly":
                        freq = "W"
                    elif report_freq == "Quarterly":
                        freq = "Q"
                    else:
                        freq = "Y"

                    st.markdown(f"### Time Series Trends ({report_freq})")
                    numeric_cols = df.select_dtypes(include='number').columns
                    resampled_df = df[numeric_cols].resample(freq).mean()

                    for col in numeric_cols:
                        st.line_chart(resampled_df[col], use_container_width=True)
                else:
                    st.warning("Time Series analysis requires a 'date' column in your data.")

            if "Correlation Heatmap" in analysis_type:
                st.markdown("### Correlation Heatmap")
                numeric_cols = df.select_dtypes(include='number')
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")
