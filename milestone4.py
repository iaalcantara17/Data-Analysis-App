import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

st.title("Data Analysis and Prediction App")

st.markdown("""
This app allows you to:
1. Upload a dataset (CSV format).
2. Select a numerical target variable.
3. Visualize your data with bar charts.
4. Train a regression model with selected features.
5. Predict the target value with new inputs.

Please follow the steps below.
""")

st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(data.head())

    if data.isnull().sum().sum() > 0:
        st.warning("Your dataset contains missing values. These will be handled during preprocessing.")


    st.header("2. Select Target Variable")
    numerical_columns = data.select_dtypes(include=np.number).columns.tolist()

    if not numerical_columns:
        st.error("No numerical columns are available in the dataset. Cannot proceed.")
    else:
        target_variable = st.selectbox(
            "Select Target Variable (must be numerical):", 
            numerical_columns
        )

        if target_variable:

            st.header("3. Visualize the Data")

            st.subheader("Average Target Value by Category")
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

            if categorical_columns:
                selected_categorical = st.radio("Select a Categorical Variable:", categorical_columns)
                if selected_categorical:
                    if data[selected_categorical].nunique() > 20:
                        st.warning("The selected variable has many categories. Consider another variable with fewer categories for clearer visualization.")
                    else:
                        avg_target = data.groupby(selected_categorical)[target_variable].mean().sort_values()
                        st.bar_chart(avg_target)
                else:
                    st.info("Please select a categorical variable.")
            else:
                st.info("No categorical columns available for category-based visualization.")

            st.subheader("Correlation Strength with Target Variable")
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.shape[1] < 2:
                st.warning("Not enough numerical columns for correlation analysis.")
            else:
                correlation = numeric_data.corr()[target_variable].drop(target_variable).abs().sort_values(ascending=False)
                if correlation.empty:
                    st.info("No other numeric columns to correlate with the target.")
                else:
                    st.bar_chart(correlation)


            st.header("4. Train the Regression Model")
            st.write("**Select features to include in the model:**")

            all_features = numerical_columns + categorical_columns
            selected_features = []
            for feat in all_features:
                if st.checkbox(feat):
                    selected_features.append(feat)

            train_button = st.button("Train Model")

            if train_button:
                if not selected_features:
                    st.error("No features selected. Please select at least one feature to train the model.")
                else:
                    X = data[selected_features]
                    y = data[target_variable]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    numeric_features = [col for col in selected_features if col in numerical_columns]
                    categorical_features = [col for col in selected_features if col in categorical_columns]

                    numeric_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ])

                    categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ])

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numeric_transformer, numeric_features),
                            ('cat', categorical_transformer, categorical_features)
                        ]
                    )

                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', GradientBoostingRegressor(random_state=42))
                    ])

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"**R² Score:** {r2:.2f}")

                    st.session_state['trained_pipeline'] = pipeline
                    st.session_state['numeric_features'] = numeric_features
                    st.session_state['categorical_features'] = categorical_features
                    st.session_state['selected_features'] = selected_features


            st.header("5. Make Predictions")
            if 'trained_pipeline' in st.session_state and 'selected_features' in st.session_state:
                feature_order = st.session_state['selected_features']
                numeric_features = st.session_state['numeric_features']
                categorical_features = st.session_state['categorical_features']
                pipeline = st.session_state['trained_pipeline']

                st.write("**Enter feature values in the following order:**")
                st.write(", ".join(feature_order))
                st.markdown("For example, if your features are `[Age, Color, Income]`, you might enter something like: `30, Red, 50000`.")

                input_features = st.text_input(
                    "Enter Feature Values (comma-separated):",
                    value="",
                    placeholder="e.g. 30, Blue, 45000"
                )

                predict_button = st.button("Predict")

                if predict_button:
                    try:
                        input_data = [val.strip() for val in input_features.split(",")]
                        if len(input_data) != len(feature_order):
                            raise ValueError("Incorrect number of input values. Please match the exact number of selected features.")

                        numeric_inputs = {}
                        categorical_inputs = {}

                        for f in numeric_features:
                            idx = feature_order.index(f)
                            try:
                                numeric_inputs[f] = float(input_data[idx])
                            except ValueError:
                                raise ValueError(f"Invalid numeric input for '{f}': '{input_data[idx]}'")

                        for f in categorical_features:
                            idx = feature_order.index(f)
                            categorical_inputs[f] = input_data[idx]

                        input_df = pd.DataFrame([{**numeric_inputs, **categorical_inputs}])

                        prediction = pipeline.predict(input_df)
                        st.write(f"**Predicted Target Value:** {prediction[0]:.2f}")

                    except ValueError as e:
                        st.error(f"Invalid input: {e}")
            else:
                st.info("Please train a model before making predictions.")
else:
    st.info("Please upload a dataset to begin.")