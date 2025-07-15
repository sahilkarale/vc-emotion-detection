import numpy as np
import pandas as pd
import pickle
import json
import logging
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# ────────────────────────────────
# Logging configuration
# ────────────────────────────────
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ────────────────────────────────
# Helper functions
# ────────────────────────────────
def load_model(file_path: str):
    """Load a pickled model from disk."""
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logger.debug("Model loaded from %s", file_path)
        return model
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading model: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data: %s", e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return evaluation metrics for a trained classifier."""
    try:
        y_pred = clf.predict(X_test)
        # Some classifiers might not implement predict_proba
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = float("nan")

        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "auc": auc,
        }
        logger.debug("Evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Write metrics dictionary to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Error while saving metrics: %s", e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Write run/model metadata to a JSON file (stub for compatibility)."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, "w") as f:
            json.dump(model_info, f, indent=4)
        logger.debug("Model info saved to %s", file_path)
    except Exception as e:
        logger.error("Error while saving model info: %s", e)
        raise


# ────────────────────────────────
# Main execution block
# ────────────────────────────────
def main():
    try:
        # Paths
        MODEL_FPATH = "./models/model.pkl"
        TEST_FPATH = "./data/processed/test_bow.csv"
        METRICS_FPATH = "reports/metrics.json"
        MODEL_INFO_FPATH = "reports/model_info.json"

        # Load model and data
        clf = load_model(MODEL_FPATH)
        test_df = load_data(TEST_FPATH)

        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        # Evaluate + persist metrics
        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, METRICS_FPATH)

        # Save a minimal model‑run stub (kept for parity with old script)
        save_model_info("N/A", MODEL_FPATH, MODEL_INFO_FPATH)

        logger.info("Model evaluation completed successfully")
        print(json.dumps(metrics, indent=4))
    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
