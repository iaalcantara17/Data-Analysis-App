# Data Analysis and Prediction App

A Streamlit web app that takes a CSV, charts it against a numeric column you choose, and trains a regression model to predict that column.

**Live demo:** https://milestone-4-data-analysis.streamlit.app

The app is on Streamlit's free tier, so it may take a moment to wake up the first time you open it.

## What it does

The app walks through five steps on one page:

1. **Upload a CSV.** It shows the first rows and warns you if the data has missing values.
2. **Pick a target.** You choose the column to predict from the numeric columns.
3. **Look at the data.** The first bar chart shows the average target value for each category of a text column you pick. It's skipped with a warning when that column has more than 20 categories. The second shows how strongly each other numeric column correlates with the target.
4. **Train a model.** You tick the columns to use as features and press Train. The app holds out 20% of the rows for testing, trains a gradient boosting regressor, and shows the R² score on the held-out rows.
5. **Predict.** You type one value per feature, separated by commas, and the app shows the predicted target. It checks that the count of values matches and that numeric features get numbers.

Preprocessing happens inside a scikit-learn pipeline. Numeric columns get mean imputation and standard scaling. Text columns get most-frequent imputation and one-hot encoding, and categories the model hasn't seen are ignored.

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn (`Pipeline`, `ColumnTransformer`, `GradientBoostingRegressor`).

## How to run it locally

You need Python 3.

```sh
pip install -r requirements.txt
streamlit run milestone4.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it and upload a CSV.

## What I'd point a reader to

- The `ColumnTransformer` in step 4 of `milestone4.py`. It sends numeric and text columns through their own imputer and encoder, so a CSV with missing values and mixed types still trains without extra cleanup.
- The prediction step reuses the trained pipeline from `st.session_state`, so new input goes through the same preprocessing as the training data.

## Limitations

- It only does regression, so the target has to be numeric.
- It uses one model with default settings and doesn't tune it.
- The target column also appears in the feature list. If you tick it, the model trains on the answer and the R² score is meaningless.
- Prediction takes one row at a time as comma-separated text, so a text value can't contain a comma.
