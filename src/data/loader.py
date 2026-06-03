from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ChurnDataLoader:
    """Loads, cleans, and splits the Telco Churn dataset.

    Accepts any CSV with a binary target column. Preprocessing is automatic:
    numeric columns → StandardScaler, categorical columns → OneHotEncoder.
    """

    def __init__(
        self,
        path: str | Path,
        target: str = "Churn",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.path = Path(path)
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> pd.DataFrame:
        logger.info("Loading dataset from %s", self.path)
        df = pd.read_csv(self.path)
        logger.info("Raw shape: %s", df.shape)
        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # TotalCharges comes as string with spaces in Telco dataset
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d rows with nulls", dropped)

        df = df.drop(columns=["customerID"], errors="ignore")

        # Encode binary target (Yes/No → 1/0)
        if df[self.target].dtype == object:
            df[self.target] = (df[self.target] == "Yes").astype(int)

        logger.info("Clean shape: %s | churn rate: %.1f%%", df.shape, df[self.target].mean() * 100)
        return df

    def get_splits(
        self, df: pd.DataFrame
    ) -> tuple:
        """Return (X_train, X_test, y_train, y_test, preprocessor).

        Preprocessor is a fitted ColumnTransformer (StandardScaler + OHE).
        Both X arrays are float32 numpy arrays ready for sklearn and PyTorch.
        """
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        num_cols = [c for c in df.columns if c != self.target and c not in cat_cols]

        X = df.drop(columns=[self.target])
        y = df[self.target].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), num_cols),
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                    cat_cols,
                ),
            ]
        )

        X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
        X_test_proc = preprocessor.transform(X_test).astype(np.float32)

        logger.info(
            "Split → train=%s test=%s | features=%d",
            X_train_proc.shape[0],
            X_test_proc.shape[0],
            X_train_proc.shape[1],
        )
        return X_train_proc, X_test_proc, y_train, y_test, preprocessor

    def dataset_info(self) -> dict:
        content = self.path.read_bytes()
        return {
            "name": self.path.name,
            "sha256": hashlib.sha256(content).hexdigest()[:12],
            "size_bytes": str(self.path.stat().st_size),
            "path": str(self.path),
        }
