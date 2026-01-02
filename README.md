![Python](https://img.shields.io/badge/Language-Python-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Status](https://img.shields.io/badge/Project-Credit_Card_Fraud_Detection-green)

# Credit Card Fraud Detection using XGBoost

An end-to-end **Machine Learning project** to detect fraudulent credit card transactions using the **XGBoost Classifier**.  
This project focuses on handling **highly imbalanced data**, improving **fraud recall**, and visualizing results using **Matplotlib**.

---

## 🚀 Features

### ✓ Data Preprocessing

- Feature scaling
- Train–test split

### ✓ Imbalanced Data Handling

- Used `scale_pos_weight` to handle rare fraud cases

### ✓ Machine Learning Model

- XGBoost Classifier
- Binary Classification (Fraud / Normal)

### ✓ Model Evaluation

- ROC-AUC Score
- Confusion Matrix
- Recall (Fraud Class)

### ✓ Visualization

- ROC Curve (Dark Theme)
- Confusion Matrix Heatmap
- Feature Importance Plot

### ✓ Model Saving

- Trained model saved as `.pkl` file

---

## ⚙️ Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost

---

## 🔧 Model Configuration

```python
Algorithm: XGBoost Classifier
Task: Binary Classification

n_estimators = 200
learning_rate = 0.05
max_depth = 5
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = 100
eval_metric = "logloss"
random_state = 42


# 📈 Model Performance

ROC-AUC: ~0.97

Fraud Recall: ~87%

Missed Fraud Cases (FN): 13



## 📊 Visualizations


All graphs were created using Matplotlib:

ROC Curve (Dark Theme)

Confusion Matrix Heatmap

Train vs Test ROC Curve

Feature Importance Bar Chart

Saved using:

plt.savefig("image.png", dpi=300, bbox_inches="tight")

##📁 Project Structure
XGboost/
│
├── _confusion_matrix.png
├── _roc.png
├── creditcard.csv        # Not pushed (large file)
├── creditcard.csv.zip
├── train_Model.ipynb
├── xgboost_fraud_model.pkl
└── README.md

##📊 Dataset Information

⚠️ Note:
The dataset is very large and exceeds GitHub’s file size limit, so it is not uploaded.

🔗 Kaggle Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place creditcard.csv in the project folder.

##▶️ How to Run
git clone https://github.com/kanha165/Credit-Card-Fraud-Detection-using-XGBoost.git
cd Credit-Card-Fraud-Detection-using-XGBoost
pip install -r requirements.txt


Open train_Model.ipynb and run cells step by step.

##🧪 Testing the Model
prob = model.predict_proba(sample)[0][1]

if prob >= 0.3:
    print("Fraud Transaction")
else:
    print("Normal Transaction")

##🧠 Core Concepts Used

Supervised Machine Learning

Boosting Algorithms

Imbalanced Data Handling

Model Evaluation Metrics

Data Visualization

Model Serialization

##🔥 Future Improvements

Streamlit deployment

Real-time fraud detection

Compare with AdaBoost & Random Forest

Apply SMOTE

Monitoring dashboard

##👤 Author

Kanha Patidar
B.Tech CSIT (5th Semester)
Chameli Devi Group of Institutions, Indore
Machine Learning Intern — Technorizen Software Solution Pvt. Ltd.

GitHub: https://github.com/kanha165

LinkedIn: https://www.linkedin.com/in/kanha-patidar-837421290/

##⭐ Acknowledgment

Dataset provided by Kaggle.
For learning, academic, and portfolio purposes.

##⭐ If you like this project, please give it a star on GitHub!
```
