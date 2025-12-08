# Technical Documentation
## Digital Twin-Driven Predictive Maintenance Model for Industrial Machines

---

## 1. Introduction
This project aims to predict machine failure using a Digital Twin–driven predictive maintenance model.  
By simulating machine operating conditions (temperature, speed, torque, tool wear) and using sensor data, the model provides **risk scores** for failure, enabling optimized maintenance scheduling and reduced production risk.  

**Key Objectives:**
- Develop interpretable baseline models (Decision Tree)  
- Train high-performance models (Random Forest, XGBoost, Neural Network)  
- Provide explainability with SHAP and PDPs  
- Allow scenario testing for maintenance decisions

**Hypotheses:**
1. Higher `tool_wear_min` increases probability of failure.  
2. Large `temperature_difference` correlates with higher failure risk.  
3. Random Forest outperforms a simple Decision Tree.  
4. A small Decision Tree (depth=2, 2 features) can provide human-understandable rules.  
5. Simulating operational scenarios changes predicted risk scores meaningfully.

---

## 2. Dataset Description
The dataset contains:

**Original Features:**
- `udi`, `product_id`, `type`, `air_temperature_k`, `process_temperature_k`,  
  `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`  

**Failure Indicators:**
- `twf`, `hdf`, `pwf`, `osf`, `rnf`  

**Engineered Features:**
- `temperature_difference`, `maschine_power`, `temp_ratio`, `power_ratio`  

**Target:**
- `machine_failure` (binary: 1 = failure, 0 = normal)

**Observations:**
- Data is stored in **DuckDB (`team_data.duckdb`)** for reproducibility  
- Raw data resides in `data/raw/`  

---

## 3. Exploratory Data Analysis (EDA)
EDA was performed to understand the distributions, correlations, and potential issues in the data:

- Histograms for numeric features (`air_temperature_k`, `torque_nm`, etc.)  
- Boxplots for detecting outliers  
- Correlation heatmaps between features and target  
- Pairplots of engineered features  
- Checking class imbalance for `machine_failure`  
- Ensuring no missing values that affect model training

---

## 4. Data Preprocessing
Preprocessing steps:

1. **Missing Value Check:** No  missing data found  
2. **Feature Encoding:** OneHotEncoder for `type`  
3. **Scaling:** Numeric features scaled for NN, optional for tree-based models  
4. **Train-Test Split:** 70/30 split, stratified on target  
5. **Removal of identifiers:** `udi`, `product_id` excluded from modeling

---

## 5. Feature Engineering
Custom features created to enhance model performance:

- `temperature_difference = process_temperature_k - air_temperature_k`  
- `maschine_power = torque_nm * rotational_speed_rpm`  
- `temp_ratio = air_temperature_k / process_temperature_k`  
- `power_ratio = maschine_power / max_power_reference`  

These features capture domain-specific insights and non-linear effects.

---

## 6. Data Pipeline
The project uses a centralized **DuckDB-based pipeline**:

1. Raw data loaded into `data/raw/`  
2. Preprocessing & feature engineering performed in Python    
3. All data written to `team_data.duckdb`  
4. Models retrieve data directly from DuckDB for training and evaluation

> This ensures reproducibility, easy updates, and consistent model input.

---

## 7. Model Training

**Models Trained:**
1. **Decision Tree (baseline)** – depth=2, 2 features for interpretability  
2. **Random Forest (final model)** – selected for best performance  
3. **XGBoost** – gradient boosting model for comparison  
4. **Neural Network (MLP)** – deep learning baseline  

**Training Steps:**
- Fit models on preprocessed & encoded features  
- Evaluate using  F1  
- Tune hyperparameters for Random Forest and XGBoost  
- Generate explainability plots (SHAP, PDP)  

---

## 8. Evaluation Metrics
Metrics used to assess model performance:

- **Precision = TP / (TP + FP)**  
- **Recall = TP / (TP + FN)**  
- **F1-Score = 2 * (Precision * Recall) / (Precision + Recall)**  
- **AUC-PR** – for imbalanced classes  

**Results Summary:**  
- Decision Tree: interpretable baseline, limited accuracy  
- Random Forest: highest F1_Score, stable results  
- XGBoost: high performance but slightly less stable  
- Neural Network: reasonable, but did not outperform trees

---
## 9. Interactive Dashboard
- Streamlit dashboard allows scenario testing  
- Adjust torque, speed, temperature to see real-time risk scores


---
## 10. Limitations
- No real timestamps → sequential/temporal modeling not possible  
- Imbalanced failure classes  
- Engineered features based on domain knowledge, not physics-based simulation  
- Small Decision Tree baseline cannot capture complex interactions

---

## 11. Future Work
- Integrate real-time sensor timestamps for temporal modeling  
- Explore LSTM / Temporal CNN models  
- Expand digital twin simulation with physics-based models  
- Deploy model as REST API for real-time prediction  


