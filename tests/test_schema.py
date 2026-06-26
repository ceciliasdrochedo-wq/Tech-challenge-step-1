import pandas as pd
import pandera as pa
import pytest
from pandera import Check, Column, DataFrameSchema

from src.config import DataConfig
from src.data.loader import ChurnDataLoader

raw_schema = DataFrameSchema(
    columns={
        "customerID": Column(str, required=True),
        "gender": Column(str, Check.isin(["Female", "Male"]), required=True),
        "SeniorCitizen": Column(int, Check.isin([0, 1]), required=True),
        "Partner": Column(str, Check.isin(["Yes", "No"]), required=True),
        "Dependents": Column(str, Check.isin(["Yes", "No"]), required=True),
        "tenure": Column(int, Check.greater_than_or_equal_to(0), required=True),
        "PhoneService": Column(str, Check.isin(["Yes", "No"]), required=True),
        "MultipleLines": Column(str, Check.isin(["No", "No phone service", "Yes"]), required=True),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"]), required=True),
        "OnlineSecurity": Column(
            str, Check.isin(["No", "No internet service", "Yes"]), required=True
        ),
        "OnlineBackup": Column(
            str, Check.isin(["No", "No internet service", "Yes"]), required=True
        ),
        "DeviceProtection": Column(
            str, Check.isin(["No", "No internet service", "Yes"]), required=True
        ),
        "TechSupport": Column(str, Check.isin(["No", "No internet service", "Yes"]), required=True),
        "StreamingTV": Column(str, Check.isin(["No", "No internet service", "Yes"]), required=True),
        "StreamingMovies": Column(
            str, Check.isin(["No", "No internet service", "Yes"]), required=True
        ),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"]), required=True
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"]), required=True),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ]
            ),
            required=True,
        ),
        "MonthlyCharges": Column(float, Check.greater_than_or_equal_to(0), required=True),
        "TotalCharges": Column(str, required=True),
        "Churn": Column(str, Check.isin(["Yes", "No"]), required=True),
    },
    strict=True,
    coerce=True,
)

cleaned_schema = DataFrameSchema(
    columns={
        "gender": Column(str, Check.isin(["Female", "Male"]), required=True),
        "SeniorCitizen": Column(int, Check.isin([0, 1]), required=True),
        "Partner": Column(str, Check.isin(["Yes", "No"]), required=True),
        "Dependents": Column(str, Check.isin(["Yes", "No"]), required=True),
        "tenure": Column(int, Check.greater_than_or_equal_to(0), required=True),
        "PhoneService": Column(str, Check.isin(["Yes", "No"]), required=True),
        "MultipleLines": Column(str, Check.isin(["No", "No phone service", "Yes"]), required=True),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"]), required=True),
        "OnlineSecurity": Column(
            str, Check.isin(["No", "No internet service", "Yes"]), required=True
        ),
        "OnlineBackup": Column(
            str, Check.isin(["No", "No internet service", "Yes"]), required=True
        ),
        "DeviceProtection": Column(
            str, Check.isin(["No", "No internet service", "Yes"]), required=True
        ),
        "TechSupport": Column(str, Check.isin(["No", "No internet service", "Yes"]), required=True),
        "StreamingTV": Column(str, Check.isin(["No", "No internet service", "Yes"]), required=True),
        "StreamingMovies": Column(
            str, Check.isin(["No", "No internet service", "Yes"]), required=True
        ),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"]), required=True
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"]), required=True),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ]
            ),
            required=True,
        ),
        "MonthlyCharges": Column(float, Check.greater_than_or_equal_to(0), required=True),
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0), required=True),
        "Churn": Column(int, Check.isin([0, 1]), required=True),
    },
    strict=True,
    coerce=True,
)


def test_raw_csv_schema():
    cfg = DataConfig()
    df_raw = pd.read_csv(cfg.path)
    validated_df = raw_schema.validate(df_raw)
    assert validated_df is not None


def test_cleaned_df_schema():
    cfg = DataConfig()
    loader = ChurnDataLoader(path=cfg.path, target=cfg.target)
    df_cleaned = loader.load()
    validated_df = cleaned_schema.validate(df_cleaned)
    assert validated_df is not None


def test_invalid_data_raises_schema_error():
    invalid_raw_df = pd.DataFrame(
        {
            "customerID": ["123"],
            "gender": ["InvalidGender"],
            "SeniorCitizen": [2],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [-5],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [50.0],
            "TotalCharges": ["50.0"],
            "Churn": ["No"],
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        raw_schema.validate(invalid_raw_df)
