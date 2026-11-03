# Multi-Algorithm Grand Tournament

A simple, clean machine learning benchmarking project that runs a tournament across **7 popular classification algorithms** on real-world data from the **UCI Machine Learning Repository**.

---

## 1. Trusted Dataset Source & Attribution

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/) (University of California, Irvine)
- **Dataset**: Bank Marketing Dataset (S. Moro, P. Cortez, P. Rita, 2014)
- **Official URL**: [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Download Link**: `https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip`
- **Total Records**: **4,521 records** (exceeds the 2,000+ record requirement)

---

## 2. Database & Table Schema

The dataset is stored in an SQLite database file (`tournament.db`) under the specified table name:

- **Database File**: `tournament.db`
- **Table Name**: `multialgo_tournament_records`

### Field Definitions:

| Field Name | Type | Description | Original Dataset Feature |
| :--- | :--- | :--- | :--- |
| `profile_id` | `TEXT PRIMARY KEY` | Unique Profile Identifier (e.g., `PROF_00001`) | Generated Sequential ID |
| `scaled_metric_alpha` | `REAL NOT NULL` | Standardized Age ($\mu = 0, \sigma = 1$) | `age` (normalized) |
| `scaled_metric_beta` | `REAL NOT NULL` | Standardized Yearly Balance ($\mu = 0, \sigma = 1$) | `balance` (normalized) |
| `scaled_metric_gamma` | `REAL NOT NULL` | Standardized Contact Duration ($\mu = 0, \sigma = 1$) | `duration` (normalized) |
| `categorical_flag_delta` | `INTEGER NOT NULL` | Housing Loan Flag (`0` = No, `1` = Yes) | `housing` |
| `evaluation_target_class` | `INTEGER NOT NULL` | Term Deposit Subscription Target (`0` = No, `1` = Yes) | `y` |

---

## 3. Contender Algorithms (7 Models)

The tournament compares 7 algorithms on an 80% train / 20% test split:
1. **Gaussian Naive Bayes**
2. **Decision Tree**
3. **Gradient Boosting**
4. **K-Nearest Neighbors (KNN)**
5. **Random Forest**
6. **Logistic Regression**
7. **Support Vector Machine (SVC)**

---

## 4. File Structure

```
multialgo_grand_tournament/
├── data_loader.py       # Downloads 4,521 UCI records and populates SQLite
├── main.py              # Runs the 7-algorithm tournament & interactive prediction
├── app.py               # Interactive web deployment dashboard (http://127.0.0.1:8080)
├── tournament.db        # SQLite database (table: multialgo_tournament_records)
├── requirements.txt     # Python package requirements
└── README.md            # Project documentation
```

---

## 5. How to Run

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Grand Tournament (Leaderboard)**:
   ```bash
   python main.py
   ```

3. **Interactive Prediction in Terminal**:
   ```bash
   python main.py --predict
   ```

4. **Launch Web Deployment Dashboard**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:8080` in your browser.
