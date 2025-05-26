import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Flexible Health Data Analyzer", layout="wide")
st.title("Flexible Health Data Analyzer")

uploaded_file = st.file_uploader(
    "Upload your health data file (CSV, Excel, TSV, JSON)", 
    type=["csv", "xlsx", "xls", "tsv", "json"]
)

def mean_abs_dev(series):
    return (series - series.mean()).abs().mean()

def descriptive_stats(df):
    desc = pd.DataFrame()
    desc['Mean'] = df.mean()
    desc['Median'] = df.median()
    desc['Std Dev'] = df.std()
    desc['Range'] = df.max() - df.min()
    desc['Mean Abs Dev'] = df.apply(mean_abs_dev)
    desc['25% Quantile'] = df.quantile(0.25)
    desc['75% Quantile'] = df.quantile(0.75)
    desc['IQR'] = desc['75% Quantile'] - desc['25% Quantile']
    desc['Coeff of Var (%)'] = (desc['Std Dev'] / desc['Mean']) * 100
    desc['Skewness'] = df.skew()
    desc['Kurtosis'] = df.kurtosis()
    return desc

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

        # البحث عن أعمدة التواريخ
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        # اختيار عمود التاريخ (إذا وجد أكثر من واحد)
        date_col = None
        if date_cols:
            date_col = st.selectbox("Select date column for resampling (if any)", date_cols)

        # اختيار فترة التقرير
        freq_map = {
            "Weekly": "W",
            "Monthly": "M",
            "Quarterly": "Q",
            "Yearly": "Y"
        }
        report_freq = st.selectbox("Select report frequency", list(freq_map.keys()))

        # اختيار نوع التحليل
        analysis_types = [
            "Descriptive Statistics",
            "Time Series",
            "Correlation Heatmap",
            "Numerical Distribution",
            "Categorical Distribution"
        ]
        selected_analysis = st.multiselect("Select analysis types", analysis_types, default=["Descriptive Statistics"])

        # اختيار الأعمدة - كل الأعمدة أو اختيار أعمدة محددة
        all_numeric_cols = df.select_dtypes(include='number').columns.tolist()
        all_cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        st.subheader("Select Columns for Analysis")

        numeric_cols = []
        cat_cols = []

        if "Descriptive Statistics" in selected_analysis or "Numerical Distribution" in selected_analysis or "Time Series" in selected_analysis or "Correlation Heatmap" in selected_analysis:
            use_all_num = st.checkbox("Use all numeric columns?", value=True)
            if use_all_num:
                numeric_cols = all_numeric_cols
            else:
                numeric_cols = st.multiselect("Select numeric columns", all_numeric_cols)

        if "Categorical Distribution" in selected_analysis:
            use_all_cat = st.checkbox("Use all categorical columns?", value=True)
            if use_all_cat:
                cat_cols = all_cat_cols
            else:
                cat_cols = st.multiselect("Select categorical columns", all_cat_cols)

        # إعادة ترتيب وتحويل التاريخ إذا اخترت التقرير الدوري والتحليل الزمني
        if date_col and report_freq and "Time Series" in selected_analysis:
            df = df.sort_values(by=date_col)
            df = df.dropna(subset=[date_col])
            df.set_index(date_col, inplace=True)
            resampled_df = df.resample(freq_map[report_freq]).mean()
        else:
            resampled_df = df

        # تحليل البيانات وعرض الرسوم

        if "Descriptive Statistics" in selected_analysis and numeric_cols:
            st.subheader("Descriptive Statistics")
            stats = descriptive_stats(df[numeric_cols])
            st.dataframe(stats)

        if "Time Series" in selected_analysis and numeric_cols and date_col:
            st.subheader(f"Time Series Analysis (Resampled {report_freq})")
            for col in numeric_cols:
                st.markdown(f"**{col}**")
                st.line_chart(resampled_df[col].dropna())

        if "Correlation Heatmap" in selected_analysis and len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            cmap = st.color_picker("Pick colormap for heatmap", "#FF5733")
            plot_heatmap(df, numeric_cols, "Correlation Heatmap", cmap)

        if "Numerical Distribution" in selected_analysis and numeric_cols:
            st.subheader("Numerical Data Distribution")
            for col in numeric_cols:
                # اختيار نوع الرسم البياني المناسب (Histogram أو Boxplot)
                chart_type = st.selectbox(f"Select chart type for numeric column '{col}'", ["Histogram", "Boxplot"], key=f"num_chart_{col}")
                # تعديل عنوان الرسم البياني
                chart_title = st.text_input(f"Chart title for '{col}'", f"{chart_type} of {col}", key=f"title_num_{col}")
                # اختيار لون الرسم البياني
                chart_color = st.color_picker(f"Pick color for '{col}' chart", "#1f77b4", key=f"color_num_{col}")
                if chart_type == "Histogram":
                    plot_histogram(df, col, chart_title, chart_color)
                else:
                    plot_boxplot(df, col, chart_title, chart_color)

        if "Categorical Distribution" in selected_analysis and cat_cols:
            st.subheader("Categorical Data Distribution")
            for col in cat_cols:
                chart_title = st.text_input(f"Chart title for categorical column '{col}'", f"Bar chart of {col}", key=f"title_cat_{col}")
                chart_color = st.color_picker(f"Pick color for '{col}' chart", "#ff7f0e", key=f"color_cat_{col}")
                plot_bar(df, col, chart_title, chart_color)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a file to start analysis.")