# 💳 Credit Card Fraud Detection using XGBoost

## 📌 Project Overview
Credit card fraud detection is a critical machine learning problem due to the highly imbalanced nature of real-world transaction data. Fraudulent transactions are extremely rare compared to normal transactions, which makes accuracy an unreliable metric.

This project implements an **end-to-end machine learning pipeline** using **XGBoost** to detect fraudulent credit card transactions. The focus is on improving **fraud recall** while maintaining strong overall performance. The project also includes **professional visualizations using Matplotlib**.

---

## 🎯 Problem Statement
- Fraud cases account for less than **0.2%** of all transactions
- Traditional models fail on imbalanced datasets
- The key goal is to **minimize missed fraud cases (False Negatives)**

---

## 💡 Solution Approach
- Used **XGBoost Classifier** for robust boosting-based learning
- Handled class imbalance using **scale_pos_weight**
- Evaluated the model using **ROC-AUC, Confusion Matrix, and Recall**
- Visualized results using **Matplotlib**
- Saved the trained model for reuse and testing

---

## 📊 Dataset Information

The dataset used in this project is the **Credit Card Fraud Detection Dataset**.

⚠️ **Note:**  
The dataset file is **very large in size**, which exceeds GitHub’s file size limit.  
Therefore, the dataset is **not uploaded to this repository**.

You can download the dataset directly from Kaggle using the link below:

🔗 **Kaggle Dataset Link:**  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Details
- Total transactions: 284,807  
- Fraud transactions: 492  
- Normal transactions: 284,315  
- Highly imbalanced dataset  
- Features `V1` to `V28` are PCA-transformed for privacy protection  

After downloading, place the `creditcard.csv` file in the project root directory before running the notebook.

---


### 🔑 Features
- `Time` – Time elapsed since the first transaction  
- `V1` to `V28` – PCA-transformed numerical features (privacy-protected)  
- `Amount` – Transaction amount  
- `Class` – Target variable  
  - `0` → Normal  
  - `1` → Fraud  

> PCA was applied to anonymize sensitive information while preserving important patterns.

---

## ⚙️ Tech Stack
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- XGBoost  

---

## 🤖 Machine Learning Model
- **Algorithm:** XGBoost Classifier  
- **Task:** Binary Classification  

### 🔧 Hyperparameters
```python
n_estimators = 200
learning_rate = 0.05
max_depth = 5
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = 100
eval_metric = "logloss"
random_state = 42


###📈 Model Evaluation
To properly evaluate performance on imbalanced data, the following metrics were used:

ROC-AUC Score

Confusion Matrix

Recall (Fraud Class)



###🔢 Results
ROC-AUC: ~0.97

Fraud Recall: ~87%

Missed Fraud Cases (FN): 13



## 📊 Visualizations
All graphs were created using Matplotlib:

ROC Curve (Dark Theme)

Confusion Matrix Heatmap

Train vs Test ROC Curve

Feature Importance Bar Chart

Images were saved using:


plt.savefig("image.png", dpi=300, bbox_inches="tight")
###🗂️ Project Structure

XGboost/
│
│
├── _confusion_matrix.png        # Confusion Matrix graph (Matplotlib)
│
├── _roc.png                     # ROC Curve graph (Matplotlib)
│
├── creditcard.csv               # Original dataset (CSV)
│
├── creditcard.csv.zip           # Dataset zip file (backup / download)
│
├── train_Model.ipynb            # Main Jupyter Notebook (model training)
│
├── xgboost_fraud_model.pkl      # Saved trained XGBoost model
│
└── README.md                    # Project documentation (GitHub)

###▶️ How to Run the Project
1️⃣ Clone the Repository

git clone https://github.com/your-username/credit-card-fraud-detection-xgboost.git
cd credit-card-fraud-detection-xgboost
2️⃣ Install Dependencies

pip install -r requirements.txt
3️⃣ Run the Notebook
Open fraud_detection.ipynb and run the cells step by step.

🧪 Testing the Model
The trained model can be tested using:

Test dataset

Manual transaction input

Custom fraud probability threshold

Example:


prob = model.predict_proba(sample)[0][1]
if prob >= 0.3:
    print("Fraud Transaction")
else:
    print("Normal Transaction")
🚀 Future Improvements
Deploy the model using Streamlit

Add real-time transaction testing

Compare with AdaBoost and Random Forest

Apply SMOTE and analyze results

###👤 Author
Kanha Patidar
B.Tech (CSIT)
Machine Learning & Data Science Enthusiast

🔗 GitHub: https://github.com/kanha165
🔗 LinkedIn: https://www.linkedin.com/in/kanha-patidar-837421290/

⭐ Acknowledgment
Dataset provided by Kaggle.
This project is intended for learning, academic, and portfolio purposes.

⭐ Support
If you like this project, please consider giving it a ⭐ on GitHub.

