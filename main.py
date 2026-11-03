"""
Multi-Algorithm Grand Tournament - Simple Runner
Benchmarks 7 classification algorithms on records from 'multialgo_tournament_records'.
"""

import os
import sys
import time
import sqlite3
import warnings
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from data_loader import prepare_and_load_data, TABLE_NAME, DB_NAME

warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_tournament_data():
    """Load records from SQLite database, auto-initialize if missing."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)
    conn = sqlite3.connect(db_path)
    
    # Check if table exists and has records
    cursor = conn.cursor()
    cursor.execute(f"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}'")
    if cursor.fetchone()[0] == 0:
        conn.close()
        prepare_and_load_data()
        conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


def train_and_benchmark():
    """Trains 7 algorithms and returns leaderboard + trained models."""
    df = get_tournament_data()

    feature_cols = ["scaled_metric_alpha", "scaled_metric_beta", "scaled_metric_gamma", "categorical_flag_delta"]
    X = df[feature_cols].values
    y = df["evaluation_target_class"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    algorithms = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in algorithms.items():
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time_ms = (time.perf_counter() - start_time) * 1000.0

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        else:
            auc = acc

        score = (0.40 * f1 + 0.40 * auc + 0.20 * acc) * 100.0

        results.append({
            "algorithm": name,
            "accuracy": round(acc * 100.0, 2),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "time_ms": round(fit_time_ms, 2),
            "tournament_score": round(score, 2)
        })
        trained_models[name] = model

    leaderboard = pd.DataFrame(results).sort_values(by="tournament_score", ascending=False).reset_index(drop=True)
    leaderboard["rank"] = leaderboard.index + 1
    
    medals = {1: "[Gold 1st]", 2: "[Silver 2nd]", 3: "[Bronze 3rd]"}
    leaderboard["award"] = leaderboard["rank"].map(lambda r: medals.get(r, f"Finalist #{r}"))

    return df, leaderboard, trained_models


def run_tournament():
    """Run tournament and display leaderboard in terminal."""
    df, leaderboard, trained_models = train_and_benchmark()

    print("=" * 75)
    print("           MULTI-ALGORITHM GRAND TOURNAMENT BENCHMARK")
    print(f" Table: {TABLE_NAME} | Total Records: {len(df)}")
    print(f" Source: UCI Machine Learning Repository (Bank Marketing Dataset)")
    print("=" * 75)

    display_cols = ["rank", "award", "algorithm", "accuracy", "f1_score", "roc_auc", "time_ms", "tournament_score"]
    headers = ["Rank", "Award", "Algorithm", "Accuracy %", "F1 Score", "ROC-AUC", "Time (ms)", "Score / 100"]
    
    print("\n" + tabulate(leaderboard[display_cols].values, headers=headers, tablefmt="fancy_grid"))
    
    champion = leaderboard.iloc[0]
    print(f"\nGRAND CHAMPION: {champion['algorithm']} ({champion['award']})")
    print(f"Score: {champion['tournament_score']} | F1: {champion['f1_score']} | ROC-AUC: {champion['roc_auc']}\n")

    # Sample Prediction Demo
    print("-" * 75)
    print("DEFAULT SAMPLE PREDICTION:")
    sample = np.array([[0.5, -0.2, 1.1, 1]])
    best_model = trained_models[champion["algorithm"]]
    pred = best_model.predict(sample)[0]
    prob = best_model.predict_proba(sample)[0, pred] if hasattr(best_model, "predict_proba") else 0.85
    label = "WILL Subscribe (Class 1)" if pred == 1 else "Will NOT Subscribe (Class 0)"
    print(f"Inputs: [alpha=0.5, beta=-0.2, gamma=1.1, delta=1]")
    print(f"Verdict: {label} (Confidence: {prob*100:.1f}%)")
    print("-" * 75)


def interactive_predict():
    """Prompts the user to enter custom inputs and predicts the outcome."""
    _, leaderboard, trained_models = train_and_benchmark()
    champion_name = leaderboard.iloc[0]["algorithm"]
    model = trained_models[champion_name]

    print("\n" + "=" * 65)
    print("         CUSTOM PROFILE PREDICTION CONSOLE")
    print(f" Model: {champion_name} (Grand Champion)")
    print("=" * 65)
    print("(Note: Standardized metrics have an average of 0.0, range usually -2.0 to +2.0)")

    try:
        val_alpha = input("Enter scaled_metric_alpha (Standardized Age) [default 0.0]: ").strip()
        alpha = float(val_alpha) if val_alpha else 0.0

        val_beta = input("Enter scaled_metric_beta (Standardized Balance) [default 0.0]: ").strip()
        beta = float(val_beta) if val_beta else 0.0

        val_gamma = input("Enter scaled_metric_gamma (Standardized Duration) [default 0.0]: ").strip()
        gamma = float(val_gamma) if val_gamma else 0.0

        val_delta = input("Enter categorical_flag_delta (Housing Loan: 0=No, 1=Yes) [default 0]: ").strip()
        delta = int(val_delta) if val_delta else 0

        vec = np.array([[alpha, beta, gamma, delta]])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0, pred] if hasattr(model, "predict_proba") else 0.85

        print("\n" + "-" * 65)
        print(f"PREDICTION RESULT:")
        if pred == 1:
            print(">> Class 1: Customer WILL subscribe to Term Deposit! <<")
        else:
            print(">> Class 0: Customer will NOT subscribe. <<")
        print(f"Confidence: {prob*100:.1f}%")
        print("-" * 65 + "\n")

    except Exception as e:
        print(f"[!] Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", "-p", action="store_true", help="Interactive prediction in terminal")
    parser.add_argument("--web", "-w", action="store_true", help="Launch web deployment dashboard")
    args = parser.parse_args()

    if args.web:
        from app import run
        run(port=8080)
    elif args.predict:
        interactive_predict()
    else:
        run_tournament()


if __name__ == "__main__":
    main()
