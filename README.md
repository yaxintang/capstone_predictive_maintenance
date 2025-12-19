# Digital Twin-Driven Predictive Maintenance Model for Industrial Machines

This project develops a Digital Twin–driven predictive maintenance model that estimates machine failure risk based on operational conditions such as temperature, torque, rotational speed, and tool wear.  
The model enables maintenance engineers and managers to simulate scenarios, understand feature influence using SHAP explanations, and optimize maintenance decisions to reduce production risk.

---

## Value Proposition

A Digital Twin–based Predictive Maintenance System that can:

- Simulate machine operating conditions (temperature, speed, torque, tool wear)
- Simulate machine failure to understand the boundary between normal and faulty operation
- Provide interpretable SHAP explanations to identify which features drive the failure risk
- Allow managers to test “What-if” scenarios  
  (e.g., *What if we increase the load? What if the temperature rises by 20°C?*)
- Reduce production risks and support smarter maintenance scheduling

---
## Project Structure
### Run the Streamlit App

- main_code folder: main coiding file "digital_twin_ml_modeling.ipynb" is stored
- Streamlit App to run: st_app.py
- Helper functions: st_function.py
- other_modeling_approachs : Folder contains different ML models
- README file : Project information
- requirements file : Python packages requirements
- techical_documentation : Technical details of project
- project_presentation/Final_PPT_DigitalTwin.pdf : Final presentation of the project

---
## Setup
Follow the steps below to run the project locally.

---

### Clone the Repository

```bash
git clone git@github.com:yaxintang/capstone_predictive_maintenance.git
cd your-repo
```
### Create and Activate a Virtual Environment

#### For macOS / Linux users

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### For Windows (PowerShell)
```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### For Windows (Git Bash)
```powershell
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Initialize the DuckDB Database (if needed)
This will create or update team_data.duckdb inside the data/ directory.

```
python src/data_pipeline/build_duckdb.py
```

### Launch the Streamlit Dashboard

The following is based on [Streamlit Main Concepts](https://docs.streamlit.io/get-started/fundamentals/main-concepts)
.

Streamlit is a Python library for creating interactive web apps for data science and machine learning.
You build apps by adding Streamlit commands to a Python script and running it with 
```bash
streamlit run app.py
```

---
##  Hypotheses

The following hypotheses guided our project:

1. Higher `tool_wear_min` values increase the probability of machine failure.
2. Large `temperature_difference` between process and ambient air correlates with higher failure risk.
3. Ensemble models like Random Forest outperform a single Decision Tree in predictive accuracy due to reduced overfitting.
4. A small, interpretable Decision Tree (2 features, max depth=2) can provide human-understandable rules for maintenance decisions.
5. Simulating different operational scenarios (e.g., increased load or speed) will meaningfully change the predicted risk scores.

---

##  Dataset Overview

The dataset contains machine condition measurements, operational metadata, and failure indicators.

### **Original Features**

- `udi` – unique identifier ranging from 1 to 10000  
- `product_id` – consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants  
- `type` – product type L, M or H (cf. product ID)  
- `air_temperature_k` – Air temperature in Kelvin, generated using a random walk process later normalized to a standard deviation of 2 K around 300 K  
- `process_temperature_k` – Process temperature in Kelvin. Generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus  
- `rotational_speed_rpm` – calculated from a power of 2860 W, overlaid with a normally distributed noise
- `torque_nm` – torque values are normally distributed around 40 Nm with a SD = 10 Nm and no negative values  
- `tool_wear_min` – Tool wear in minutes.The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process  

### **Original Failure Signals (Binary Labels)**  
(Used to engineer the final target)

- `twf` – tool wear failure: the tool will be replaced of fail at a randomly selected tool wear time between 200 - 240 mins 
- `hdf` – Heat Dissipation Failure. 
Heat dissipation causes a process failure, if the 
difference between air- and process temperature is 
below 8.6 K and the tool’s rotational speed is below 
1380 rpm. This is the case for 115 data points.   
- `pwf` – Power Failure. The product of torque and rotational speed (in rad/s) 
equals the power required for the process. If this 
power is below 3500 W or above 9000 W, the 
process fails, which is the case 95 times in our 
dataset.   
- `osf` – Overstrain Failure. If the product of tool wear and torque exceeds 
11,000 minNm for the L product variant  
(12,000 for M, 13,000 for H), the process fails due 
to overstrain. This is true for 98 datapoints.  
- `rnf` – Random Failure. each process has a chance of 0,1 % to fail regardless 
of its process parameters. This is the case for 19 
datapoints, more frequent than could be expected 
for 10,000 datapoints in our dataset.   

### **Engineered Features**

- `temperature_difference` – Δ between process & air temperature  
- `maschine_power` – Torque × rotational speed  
- `temp_ratio` – Ratio of air/process temperature  
- `power_ratio` – Ratio of machine power to designed performance  

### **Target**

- `machine_failure` – Indicates, whether the machine has failed in this particular datapoint for any of the following failure modes

---
##  Data Pipeline

The data pipeline ensures reproducible and efficient data handling for all machine learning models:

1. **Raw Data Ingestion:**  
   All raw sensor and machine operation data are stored in `data/raw/` (CSV, etc.).


2. **Preprocessing & Feature Engineering:**  
   - Cleaning missing values  
   - Creating engineered features such as `temperature_difference`, `maschine_power`, `power_ratio`  
   - Encoding categorical features using OneHotEncoder  
   - Scaling numeric features

3. **Processed Data Storage:**  
   Final cleaned and processed datasets are stored in `data/processed/`. (leave as future work)

4. **Centralized Database (DuckDB):**  
   All  data are written to `data/team_data.duckdb`.  
   Machine learning models retrieve data directly from this database for training and evaluation.

> This setup allows reproducibility, smooth experimentation, and easy updates for multiple models.

### Data Pipeline Overview

The following diagram illustrates the ETL pipeline used in this project:

![Data Pipeline Diagram](image-9.png)

**Flow:**  
**Raw Data → Extract → Transform → Load → DuckDB**


---

##  Project Structure


---

##  Methods Used

- Exploratory Data Analysis (EDA)  
- Missing value analysis  
- Feature Engineering  
- Categorical encoding using OneHotEncoder  
- Scaling numeric features  
- Train-test split  
- Machine learning model: Random forest Classifier, XGBoostClassifier, Neural network and Decision Tree als Baseline  
- Optimized using  **F1_score**
- Model explainability: SHAP  
- Partial Dependence Plots (PDP)  

---

## Model Description

Four different machine learning models were developed and evaluated to predict machine failure risk.  
The goal was to compare baseline, tree-based, and neural network approaches to identify the best-performing algorithm for predictive maintenance.
We intentionally started with a **human-interpretable baseline model** before moving to more complex algorithms.  
This ensures transparent decision-making in critical industrial applications.

### **1. Baseline Model: Interpretable Decision Tree (Depth = 2, 2 Features)**

To establish a transparent benchmark, we trained a very small **Decision Tree**:

- **max_depth = 2**  
- **only 2 features** were used (the most influential according to initial analysis):
  - `torque_nm`
  - `rotational_speed_rpm`

 **Purpose:**  
Provide a *human-level interpretable model* that maintenance engineers can understand without ML background.

🛠 Why this approach:
- Each decision path corresponds to a simple rule  
  → e.g., *“If torque > X and speed > Y, risk increases.”*  
- Allows domain experts to validate whether the splits make physical sense  
- Serves as a logic-based baseline to compare more complex models  

Although accuracy was limited, this model provides:
- maximum interpretability  
- insights into fundamental feature thresholds  
- a sanity check before moving on to advanced models

![alt text](image.png)

---

### **2. Random Forest (Final Model)**
Random Forest was the **best-performing model** and became the final choice.

Reasons for superior performance:
- strong generalization through bagging  
- low variance compared to a single tree  
- robustness to noisy sensor data  
- excellent performance in F1-score  

Random Forest was the most stable and reliable across all validation folds.

---

### **3. XGBoost**
XGBoost offered strong performance and captured non-linear relationships well.  
However:

- required more tuning  
- slightly less stable than RandomForest on imbalanced data  
- more sensitive to hyperparameters  

It performed close to RandomForest but did not surpass it consistently.

---

### **4. Neural Network (MLP)**
A Multi-Layer Perceptron (MLP) was trained as a deep learning baseline.

Key characteristics:
- Able to learn non-linear relationships  
- Requires scaling + careful tuning  
- Sensitive to noise and imbalanced data
- required more data to generalize    
  
The NN performed reasonably well but did not outperform the tree-based models,in precision-recall metrics.

It served as a useful comparison but was not selected as the final model.

---

### **Final Choice**
The **Random Forest** was selected as the final predictive maintenance model due to:

- best f1_score  
- high robustness  
- interpretable feature importance  
- strong performance without overfitting  

The simple interpretable Decision Tree remains a valuable reference model to ensure transparency and trust. 
  

The model predicts a **risk score** representing the probability of machine failure under given operating conditions.

---

## Evaluation Metrics

The target of the model is to correctly classify whether there will be a failure, hence it is the target (failure = 1).
We want the model to:
- maximize identification of failures by minimizing "false negatives" (recall)
- maximize tool run-time by minimizing "false positives" (precision)
We will therefore go with the **F1 score** as it provides a balance between recall and precision.

Key metrics used:

- **AUC-PR** – recommended for imbalanced classification   
- **F1-Score**  

-> **Precision (Positive Predictive Value)**

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Where:  
- $TP$ = True Positives  
- $FP$ = False Positives



-> **Recall (Sensitivity / True Positive Rate)**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Where:  
- $FN$ = False Negatives


-> **F1-Score**

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---
##  Model Explainability

Understanding why the predictive maintenance model triggers a risk alert is critical for engineers, operators, and managers.
To ensure transparency, interpretability, and trust, several explainability techniques were applied.

- Global Feature Importance
- SHAP Summary Plot
- SHAP Dependence
- Partial Dependence Plots
- Interpretation of Baseline Decision Tree (human-level)
---

## 📊 Results / Visualizations

The following visualizations summarize the performance and explainability of the predictive maintenance models:

---

### **Model Performance**

**Model Comparison Table / Barplot**  
- Compares F1-score for all four models:  
  - Decision Tree (baseline)  
  - Random Forest (final)  
  - XGBoost  
  - Neural Network  

  ![alt text](image-6.png)

The chart shows the **F1 score for the failure class** across different models.  
Due to the **strong class imbalance**, F1 score is used instead of accuracy, as it better reflects performance on rare failure events.

| Model            | F1 Score (Failure Class) |
|------------------|--------------------------|
| Baseline Model   | 31%                      |
| Decision Tree    | 65%                      |
| MLP              | 81%                      |
| XGBoost          | 82%                      |
| Random Forest    | **87%**                  |

**Observations:**
- The baseline model performs poorly on failure detection.
- Tree-based and neural models significantly improve performance.
- **Random Forest** achieves the best results, indicating that ensemble methods are well suited for this predictive maintenance task.

**Conclusion:**  

The Random Forest model was selected for further experiments and explainability analysis.

---

### Interpretable Baseline: Simple Decision Tree

**Decision Tree (Depth=2, 2 Features)**  

Before training complex models, an intentionally simple Decision Tree (depth=2) was used as a baseline.

Purpose:
- Provide human-readable logic
- Validate key failure relationships
- Establish an interpretable reference model

Because it's extremely shallow (only two levels), it can be visualized and explained to non-technical stakeholders.  

- Example:  
  - *If `torque_nm > 150` and `rotational_speed_rpm > 5000` → higher failure risk.*  
- Demonstrates transparency and explains basic decision logic.

![alt text](image-5.png)
---

### **Feature Importance & Explainability**

We first analyze which features have the strongest influence on failure prediction across the entire dataset.
This helps identify dominant drivers of machine health.

**Random Forest Feature Importance (Barplot)**  
- Shows which features most influence predictions.  
- Example features: `torque_nm`, `rotational_speed_rpm`, `tool_wear_min`, `temperature_difference`.
![alt text](image-10.png)

### SHAP Explainability (Shapley Values)
To achieve model-agnostic interpretability, we use SHAP, which assigns each feature a contribution to each individual prediction.

**SHAP Summary Plot**  
Shows the overall impact and direction of each feature across all predictions.

Common insights include:
- Higher torque_nm increases failure likelihood
- Increasing tool_wear_min strongly contributes to failures
- Rotational speed has non-linear effects captured by the model
![alt text](image-7.png)

**SHAP Force Plot**  
Visualize how a feature affects the prediction while holding all other features constant.
Useful to understand interactions between variables (e.g., torque × speed).

![alt text](image-8.png)

**Partial Dependence Plots (PDPs)** 
- Visualize the average marginal effect of a feature on the predicted failure probability.  
- Useful for “What-if” analysis: e.g., *how does increasing torque affect risk?*

These plots help engineers answer:
“What happens if I increase torque by 10%?”
“How sensitive is the model to temperature changes?”

![alt text](image-4.png)


---

### **Notes**

- Only key plots are included in the README for clarity.  
- Detailed plots (EDA plots for all features) are included in the **notebooks** 
- Visualizations support **interpretability, scenario simulation, and model validation** for industrial maintenance decisions.

---
## Interactive Dashboard

**Streamlit dashboard implemented**:

- Modify input features (torque, speed, temperature,...)  
- Visualize predicted failure risk scores  
- Helps engineers and managers make informed decisions about running conditions before real use of machine

---

## Future Work

- Integrate real-time sensor timestamps for temporal modeling  
- Explore LSTM / Temporal CNN models for sequential prediction  
- Expand digital twin simulation with physics-based models
- Having try with the real-world dataset
---


