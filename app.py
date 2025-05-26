import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.impute import SimpleImputer

st.markdown("""
<style>
/* Set background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    color: #102a43;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Style sidebar */
[data-testid="stSidebar"] {
    background-color: #334e68;
    color: white;
    font-size: 18px;
    padding: 20px 15px;
    border-radius: 0 15px 15px 0;
}

/* Sidebar header */
[data-testid="stSidebar"] > div:first-child {
    font-weight: 700;
    font-size: 24px;
    margin-bottom: 1rem;
    color: #f0f4f8;
}

/* Style main header */
h1, h2, h3 {
    color: #334e68;
    font-weight: 700;
}

/* Style buttons */
.stButton > button {
    background: #627d98;
    color: white;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    transition: background 0.3s ease;
}
.stButton > button:hover {
    background: #486581;
    color: #f0f4f8;
}

/* Input fields */
input[type="number"], input[type="text"], select, textarea {
    border-radius: 8px;
    border: 1.5px solid #627d98;
    padding: 8px;
    font-size: 16px;
}

/* Dataframe/table */
[data-testid="stTable"] {
    border-radius: 10px;
    border: 1.5px solid #334e68;
}

/* Plots container */
[data-testid="stPlotContainer"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 2px 2px 6px rgba(51, 78, 104, 0.15);
}

/* Footer */
footer {
    visibility: hidden;
}
.custom-label {
    color: #000000;  /* black color */
    font-weight: 700;
    font-size: 18px;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔬 Data Transformation App")
st.markdown("Upload a CSV file to begin, apply transformations, and export the results.")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 Data Description")
    st.write(df.describe(include='all'))

    # --- Manual Imputation UI ---
    st.subheader("🧼 Handle Missing Values (Imputation)")

    numeric_df = df.select_dtypes(include=[np.number])
    numeric_cols_with_na = numeric_df.columns[numeric_df.isnull().any()].tolist()

    categorical_df = df.select_dtypes(exclude=[np.number])
    categorical_cols_with_na = categorical_df.columns[categorical_df.isnull().any()].tolist()

    cols_with_na = numeric_cols_with_na + categorical_cols_with_na

    if len(cols_with_na) > 0:
        st.write("Missing values detected in columns:")
        st.dataframe(df[cols_with_na].isnull().sum())

        impute_cols = st.multiselect(
            "Select columns to impute",
            cols_with_na,
            default=cols_with_na
        )

        impute_strategy = st.selectbox(
            "Select imputation strategy",
            ['mean', 'median', 'most_frequent', 'constant']
        )

        fill_value = None
        if impute_strategy == 'constant':
            fill_value = st.text_input("Enter constant value (will be used to fill missing values)", value="0")

        if st.button("Impute Missing Values"):
            if impute_cols:
                for col in impute_cols:
                    if col in numeric_cols_with_na:
                        imputer = SimpleImputer(
                            strategy=impute_strategy,
                            fill_value=fill_value if impute_strategy == 'constant' else None
                        )
                    else:
                        strategy = impute_strategy if impute_strategy in ['most_frequent', 'constant'] else 'most_frequent'
                        imputer = SimpleImputer(
                            strategy=strategy,
                            fill_value=fill_value if strategy == 'constant' else None
                        )
                    df[[col]] = imputer.fit_transform(df[[col]])
                st.success("Imputation completed.")
                st.dataframe(df.head())
            else:
                st.warning("Select at least one column to impute.")
    else:
        st.success("No missing values detected!")

    # --- Transformation Section ---
    st.subheader("🔁 Apply Data Transformations")
    transform_cols = st.multiselect(
        "Select numeric columns for transformation",
        df.select_dtypes(include=[np.number]).columns.tolist()
    )

    transformation = st.selectbox(
        "Choose a transformation",
        ["None", "Log", "Reciprocal", "Square Root", "Power (Box-Cox/Yeo-Johnson)"]
    )

    if transformation != "None" and transform_cols:
        df_transformed = df.copy()
        try:
            if transformation == "Log":
                transformer = FunctionTransformer(np.log1p, validate=True)
            elif transformation == "Reciprocal":
                transformer = FunctionTransformer(lambda x: 1 / (x + 1e-6), validate=True)
            elif transformation == "Square Root":
                transformer = FunctionTransformer(np.sqrt, validate=True)
            elif transformation == "Power (Box-Cox/Yeo-Johnson)":
                transformer = PowerTransformer(method='yeo-johnson')
            else:
                transformer = None

            if transformer:
                df_transformed[transform_cols] = transformer.fit_transform(df[transform_cols])
                df = df_transformed
                st.success(f"Applied {transformation} transformation.")
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error applying transformation: {e}")

    # --- Plots ---
    st.subheader("📉 Distribution & Q-Q Plots")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        plot_col = st.selectbox("Select column to visualize", numeric_columns)
        if plot_col:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[plot_col], kde=True, ax=axes[0])
            axes[0].set_title(f"Distribution of {plot_col}")

            stats.probplot(df[plot_col], dist="norm", plot=axes[1])
            axes[1].set_title("Q-Q Plot")

            st.pyplot(fig)
    else:
        st.info("No numeric columns available for plotting.")

    # --- Download Data ---
    st.subheader("⬇️ Export Transformed Data")

    def convert_df_to_csv(df_):
        return df_.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df)

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name='transformed_data.csv',
        mime='text/csv'
    )

else:
    st.info("Please upload a CSV file to begin.")
