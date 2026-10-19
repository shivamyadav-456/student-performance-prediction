# Student Performance Prediction System Using Machine Learning

An educational Machine Learning web application designed to forecast a secondary school student's academic performance category (**Low**, **Medium**, or **High Performance**) using demographic, socioeconomic, behavioral, and academic features from the UCI Student Performance Dataset.

---

## 📌 Project Overview

- **Target Categories**:
  - 🔴 **Low Performance**: Final grade $G3 < 10$ ($0 - 9$ on Portuguese 20-point scale; failing mark).
  - 🔵 **Medium Performance**: Final grade $10 \le G3 \le 13$ (satisfactory passing standard).
  - 🟢 **High Performance**: Final grade $14 \le G3 \le 20$ (good to excellent performance).
- **Core Algorithm**: Scikit-learn **Random Forest Classifier** with balanced class weighting.
- **Leakage Prevention**: The final grade column ($G3$) is strictly used to construct the target label and dropped before feature transformation. Earlier grades ($G1$ and $G2$) are documented and utilized as key period indicators.
- **Interface**: Interactive multi-page **Streamlit** dashboard featuring live inference, class probability breakdowns, baseline comparisons, confusion matrix heatmaps, and dataset exploration tools.

---

## 🛠️ Technology Stack

- **Language**: Python 3.10+ (tested on Python 3.13)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, ColumnTransformer, OneHotEncoder, StandardScaler)
- **Model Serialization**: Joblib
- **Web Interface**: Streamlit
- **Visualizations**: Plotly Express & Graph Objects

---

## 📁 Project Structure

```text
ML_project/
├── data/
│   ├── student-mat.csv         # UCI Mathematics dataset (semicolon-delimited)
│   ├── student-por.csv         # UCI Portuguese dataset
│   └── student.txt             # Dataset attributes documentation
├── models/
│   ├── student_rf_model.joblib # Saved trained model & preprocessing pipeline
│   └── model_metadata.json     # Saved evaluation metrics and schema
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Automatic UCI download, loading, & labeling
│   ├── preprocessor.py         # ColumnTransformer pipelines & metadata schema
│   ├── train.py                # Model training, baselines, and evaluation
│   └── predict.py              # Inference logic, validation, & interpretation
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # Automated unit and integration tests
├── app.py                      # Interactive Streamlit Web Application
├── train_model.py              # CLI script to train and evaluate the model
├── requirements.txt            # Project dependencies
└── README.md                   # Execution and setup instructions
```

---

## 🚀 Setup & Execution Instructions (Windows)

### 1. (Optional) Create and Activate Virtual Environment
Open PowerShell or Command Prompt in the `ML_project` directory:
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
Install all required libraries:
```powershell
pip install -r requirements.txt
```

### 3. Dataset Setup
The dataset is downloaded automatically if missing. If you need manual placement:
1. Download `student+performance.zip` from:  
   https://archive.ics.uci.edu/dataset/320/student+performance
2. Extract `student-mat.csv` into the `data/` folder:
   ```text
   ML_project/data/student-mat.csv
   ```

### 4. Train and Evaluate the Model
Train the Random Forest Classifier, compare with baselines, and save artifacts:
```powershell
python train_model.py
```

### 5. Run Automated Tests
Verify pipeline integrity, data loading, boundary rules, and predictions:
```powershell
python -m unittest tests/test_pipeline.py -v
```

### 6. Launch the Streamlit Web Application
Start the interactive dashboard:
```powershell
streamlit run app.py
```
Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 📊 Empirical Evaluation Results

Evaluated on a held-out test set ($20\%$ stratified split, $79$ test records):

| Metric | Random Forest (Primary) | Logistic Regression Baseline | Majority Dummy Baseline |
| :--- | :---: | :---: | :---: |
| **Test Accuracy (79 samples)** | **83.54%** | 87.34% | 41.77% |
| **5-Fold Cross-Validation** | **84.80% ± 3.74%** | 82.91% ± 2.91% | 41.77% |
| **Macro Precision** | **0.8387** | 0.8711 | 0.1392 |
| **Macro Recall** | **0.8485** | 0.8808 | 0.3333 |
| **Macro F1-Score** | **0.8428** | 0.8756 | 0.1964 |
| **Weighted F1-Score**| **0.8343** | 0.8732 | 0.2464 |

> **Statistical Note on Model Comparison**: On 5-Fold Cross-Validation across the training data, **Random Forest achieves higher accuracy (84.80% vs 82.91%)** and higher macro F1. On the single 79-sample held-out test split, the difference (87.34% vs 83.54%) corresponds to **only 3 individual students**, well within 1 standard deviation of sampling noise. Furthermore, Random Forest achieves **95.0% recall on High Performance students** (vs 85.0% for Logistic Regression).

### Classification Report (Random Forest)
- **Low Performance** ($G3 < 10$): Precision: `0.84`, Recall: `0.81`, F1: `0.82` ($n=26$)
- **Medium Performance** ($10 \le G3 \le 13$): Precision: `0.81`, Recall: `0.79`, F1: `0.80` ($n=33$)
- **High Performance** ($14 \le G3 \le 20$): Precision: `0.86`, Recall: `0.95`, F1: `0.90` ($n=20$)

### Top Influential Features
1. `G2` (Period 2 Grade) - $35.98\%$ importance
2. `G1` (Period 1 Grade) - $23.20\%$ importance
3. `absences` (School absences) - $3.42\%$ importance
4. `failures` (Past course failures) - $3.22\%$ importance
5. `goout` (Socializing frequency) - $2.59\%$ importance

---

## ❓ Common Troubleshooting Steps

1. **`FileNotFoundError: Trained model artifact not found`**:
   - Cause: The application was launched before running the training script.
   - Solution: Run `python train_model.py` to train and save `models/student_rf_model.joblib`.

2. **Dataset CSV parsing error or 1 column detected**:
   - Cause: UCI dataset uses semicolon (`;`) delimiters rather than commas (`,`).
   - Solution: The application and scripts explicitly use `sep=';'`. Ensure your CSV has not been re-saved with a different delimiter.

3. **Port 8501 is already in use**:
   - Solution: Run on a different port:
     ```powershell
     streamlit run app.py --server.port 8502
     ```

---

## ☁️ Streamlit Community Cloud Deployment Guide

To deploy this application to **Streamlit Community Cloud**:

1. **Repository Structure Verification:**
   Ensure the repository contains:
   ```text
   app.py
   requirements.txt
   .gitignore
   data/student-mat.csv
   models/student_rf_model.joblib
   models/model_metadata.json
   src/
   ```
   *(All internal temporary backup folders like `backup_portugal/` and `backup_indian/` are ignored via `.gitignore`.)*

2. **Deploy on Streamlit Cloud:**
   - Log into [share.streamlit.io](https://share.streamlit.io).
   - Select **New app**.
   - Choose your GitHub repository, branch (`main` or `master`), and set **Main file path** to `app.py`.
   - Click **Deploy!**.

3. **Self-Contained Execution:**
   - The application does not require any local absolute paths or external database connections.
   - If `student-mat.csv` is ever deleted, the built-in data loader automatically downloads and extracts it on the fly.

---

## ⚖️ Ethical & Educational Disclaimer

This prediction system is an **educational decision support aid** built for academic analysis. Output probabilities reflect statistical patterns in historical training data and are **not guarantees of student capability or future performance**. The tool should be used to support students constructively through early academic tutoring and counseling rather than for punitive or exclusionary decisions.
