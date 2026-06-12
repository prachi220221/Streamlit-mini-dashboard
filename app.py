import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

from openpyxl import Workbook

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)

st.title("Titanic Survival Prediction Dashboard")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = sns.load_dataset("titanic")

# Add Serial Number
df.insert(0, "Serial_Number", range(1, len(df)+1))

st.subheader("Raw Titanic Dataset")
st.dataframe(df.head())

# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------

data = df.copy()

# Target Variable
target = "survived"

# Remove rows with missing target
data = data.dropna(subset=[target])

# Separate predictors and target
X = data.drop(columns=[target])

y = data[target]

# Remove serial number from training
X_model = X.drop(columns=["Serial_Number"])

# Separate numeric and categorical
num_cols = X_model.select_dtypes(include=np.number).columns
cat_cols = X_model.select_dtypes(exclude=np.number).columns

# Missing value treatment
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_model[num_cols] = num_imputer.fit_transform(X_model[num_cols])
X_model[cat_cols] = cat_imputer.fit_transform(X_model[cat_cols])

# Encode categorical variables
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    X_model[col] = le.fit_transform(X_model[col].astype(str))
    le_dict[col] = le

# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# MODEL
# -------------------------------------------------

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# -------------------------------------------------
# METRICS
# -------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

fpr_value = FP / (FP + TN)

tpr_value = TP / (TP + FN)

roc_auc = roc_auc_score(y_test, y_prob)

fpr_curve, tpr_curve, thresholds = roc_curve(y_test, y_prob)

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------

st.subheader("Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    columns=["Predicted 0","Predicted 1"],
    index=["Actual 0","Actual 1"]
)

st.dataframe(cm_df)

metrics_df = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "FPR",
        "TPR",
        "ROC AUC"
    ],
    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        fpr_value,
        tpr_value,
        roc_auc
    ]
})

st.subheader("Model Evaluation")

st.dataframe(metrics_df)

# -------------------------------------------------
# ROC CURVE
# -------------------------------------------------

roc_df = pd.DataFrame({
    "FPR":fpr_curve,
    "TPR":tpr_curve,
    "Threshold":thresholds
})

st.subheader("ROC Curve Data")

st.dataframe(roc_df.head())

st.line_chart(
    roc_df.set_index("FPR")["TPR"]
)

# -------------------------------------------------
# EXPORT TO EXCEL
# -------------------------------------------------

if st.button("Generate Excel Report"):

    train_export = X_train.copy()
    train_export["survived"] = y_train.values

    test_export = X_test.copy()
    test_export["survived"] = y_test.values

    # Add Serial Numbers for train/test
    train_export.insert(
        0,
        "Serial_Number",
        range(1, len(train_export)+1)
    )

    test_export.insert(
        0,
        "Serial_Number",
        range(1, len(test_export)+1)
    )

    with pd.ExcelWriter(
        "Titanic_Model_Report.xlsx",
        engine="openpyxl"
    ) as writer:

        # Sheet1
        df.to_excel(
            writer,
            sheet_name="Raw_Data",
            index=False
        )

        # Sheet2
        train_export.to_excel(
            writer,
            sheet_name="Train_Data",
            index=False
        )

        # Sheet3
        test_export.to_excel(
            writer,
            sheet_name="Test_Data",
            index=False
        )

        # Sheet4
        cm_df.to_excel(
            writer,
            sheet_name="Confusion_Matrix"
        )

        # Sheet5
        metrics_df.to_excel(
            writer,
            sheet_name="Model_Evaluation",
            index=False
        )

        # Sheet6
        roc_df.to_excel(
            writer,
            sheet_name="ROC_AUC_Curve",
            index=False
        )

    with open(
        "Titanic_Model_Report.xlsx",
        "rb"
    ) as file:

        st.download_button(
            label="Download Excel Report",
            data=file,
            file_name="Titanic_Model_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.success("Excel Report Generated Successfully!")