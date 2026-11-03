"""
Multi-Algorithm Grand Tournament - Interactive Web Deployment
Runs a zero-dependency local web app on http://127.0.0.1:8080
"""

import json
import sqlite3
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Pre-train models once on startup
DB_PATH = "tournament.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM multialgo_tournament_records", conn)
conn.close()

features = ["scaled_metric_alpha", "scaled_metric_beta", "scaled_metric_gamma", "categorical_flag_delta"]
X = df[features].values
y = df["evaluation_target_class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

MODELS = {
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

LEADERBOARD = []
for name, m in MODELS.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    probs = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else preds
    acc = round(accuracy_score(y_test, preds) * 100, 2)
    f1 = round(f1_score(y_test, preds, zero_division=0), 4)
    auc = round(roc_auc_score(y_test, probs), 4)
    score = round((0.4 * f1 + 0.4 * auc + 0.2 * (acc / 100.0)) * 100, 2)
    LEADERBOARD.append({"algorithm": name, "accuracy": acc, "f1": f1, "auc": auc, "score": score})

LEADERBOARD.sort(key=lambda x: x["score"], reverse=True)
for i, item in enumerate(LEADERBOARD):
    item["rank"] = i + 1
    item["medal"] = "🥇 Gold" if i == 0 else ("🥈 Silver" if i == 1 else ("🥉 Bronze" if i == 2 else f"#{i+1}"))

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Multi-Algorithm Grand Tournament Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #334155; }
        h1 { margin: 0 0 8px 0; color: #38bdf8; font-size: 26px; }
        .subtitle { color: #94a3b8; font-size: 14px; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 24px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; }
        th { background: #0f172a; color: #94a3b8; text-transform: uppercase; font-size: 11px; }
        tr:hover td { background: #334155; }
        .form-group { margin-bottom: 12px; }
        label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 4px; font-weight: 600; }
        input, select { width: 100%; padding: 10px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; color: #fff; box-sizing: border-box; }
        button { width: 100%; background: #38bdf8; color: #0f172a; font-weight: 700; border: none; padding: 12px; border-radius: 6px; cursor: pointer; font-size: 14px; margin-top: 8px; }
        button:hover { background: #7dd3fc; }
        .result-box { text-align: center; padding: 20px; border-radius: 8px; margin-top: 14px; }
        .res-pos { background: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e; color: #4ade80; }
        .res-neg { background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #f87171; }
        .confidence { font-size: 32px; font-weight: 800; margin: 8px 0; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🏆 Multi-Algorithm Grand Tournament</h1>
        <div class="subtitle">Benchmarking 7 Classification Algorithms on 4,521 Records (UCI Repository)</div>
    </div>

    <!-- Leaderboard -->
    <div class="card">
        <h2 style="font-size: 18px; margin-bottom: 14px; color: #38bdf8;">🏁 Tournament Leaderboard</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Award</th>
                    <th>Algorithm</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                    <th>ROC-AUC</th>
                    <th>Grand Score</th>
                </tr>
            </thead>
            <tbody id="lb-body"></tbody>
        </table>
    </div>

    <!-- Live Predictor -->
    <div class="grid">
        <div class="card">
            <h2 style="font-size: 18px; margin-bottom: 14px; color: #38bdf8;">🔮 Live Prediction Studio</h2>
            <div class="form-group">
                <label>scaled_metric_alpha (Standardized Age: avg is 0.0)</label>
                <input type="number" id="alpha" step="0.1" value="0.5">
            </div>
            <div class="form-group">
                <label>scaled_metric_beta (Standardized Balance: avg is 0.0)</label>
                <input type="number" id="beta" step="0.1" value="-0.2">
            </div>
            <div class="form-group">
                <label>scaled_metric_gamma (Standardized Duration: avg is 0.0)</label>
                <input type="number" id="gamma" step="0.1" value="1.1">
            </div>
            <div class="form-group">
                <label>categorical_flag_delta (Housing Loan: 0=No, 1=Yes)</label>
                <select id="delta">
                    <option value="0">0 - No Housing Loan</option>
                    <option value="1" selected>1 - Has Housing Loan</option>
                </select>
            </div>
            <button onclick="doPredict()">⚡ Predict with Champion Model</button>
        </div>

        <div class="card" style="display: flex; flex-direction: column; justify-content: center;">
            <h2 style="font-size: 18px; margin-bottom: 6px; color: #38bdf8; text-align: center;">Prediction Result</h2>
            <div id="result-box" class="result-box res-neg">
                <div style="font-size: 13px; text-transform: uppercase;">Predicted Class</div>
                <div class="confidence" id="res-class">Class 0</div>
                <div id="res-desc" style="font-size: 15px; font-weight: 600;">Customer will NOT subscribe</div>
                <div id="res-prob" style="font-size: 13px; margin-top: 6px; opacity: 0.85;">Confidence: 87.0%</div>
            </div>
        </div>
    </div>
</div>

<script>
    const lbData = """ + json.dumps(LEADERBOARD) + """;
    const tbody = document.getElementById('lb-body');
    tbody.innerHTML = lbData.map(r => `
        <tr>
            <td><strong>#${r.rank}</strong></td>
            <td>${r.medal}</td>
            <td><strong>${r.algorithm}</strong></td>
            <td>${r.accuracy}%</td>
            <td>${r.f1}</td>
            <td>${r.auc}</td>
            <td><strong style="color: #38bdf8;">${r.score}</strong></td>
        </tr>
    `).join('');

    async function doPredict() {
        const alpha = parseFloat(document.getElementById('alpha').value) || 0;
        const beta = parseFloat(document.getElementById('beta').value) || 0;
        const gamma = parseFloat(document.getElementById('gamma').value) || 0;
        const delta = parseInt(document.getElementById('delta').value) || 0;

        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ alpha, beta, gamma, delta })
        });
        const data = await res.json();

        const box = document.getElementById('result-box');
        const cls = document.getElementById('res-class');
        const desc = document.getElementById('res-desc');
        const prob = document.getElementById('res-prob');

        if (data.prediction === 1) {
            box.className = 'result-box res-pos';
            cls.innerText = 'Class 1';
            desc.innerText = 'Customer WILL Subscribe to Term Deposit!';
        } else {
            box.className = 'result-box res-neg';
            cls.innerText = 'Class 0';
            desc.innerText = 'Customer will NOT subscribe.';
        }
        prob.innerText = 'Model Confidence: ' + data.confidence + '%';
    }
</script>
</body>
</html>
"""


class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        data = json.loads(body)

        alpha = float(data.get("alpha", 0))
        beta = float(data.get("beta", 0))
        gamma = float(data.get("gamma", 0))
        delta = int(data.get("delta", 0))

        # Champion model is rank 1
        champion_name = LEADERBOARD[0]["algorithm"]
        champion_model = MODELS[champion_name]

        vec = np.array([[alpha, beta, gamma, delta]])
        pred = int(champion_model.predict(vec)[0])
        prob = champion_model.predict_proba(vec)[0, pred] if hasattr(champion_model, "predict_proba") else 0.85

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {
            "algorithm": champion_name,
            "prediction": pred,
            "confidence": round(float(prob * 100), 1)
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))


def run(port=8080):
    server = HTTPServer(("127.0.0.1", port), WebHandler)
    print("=" * 70)
    print(f" Web Deployment running at: http://127.0.0.1:{port}")
    print(" Open the above link in your browser to test predictions visually!")
    print(" Press Ctrl+C in terminal to stop.")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    run(port=8080)
