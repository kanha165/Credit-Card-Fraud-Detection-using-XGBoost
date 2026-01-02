# 💳 Credit Card Fraud Detection using XGBoost

## 📌 Project Overview
This project implements a machine learning model to detect fraudulent credit card transactions using **XGBoost**.
Due to the highly imbalanced nature of the dataset, special techniques are used to improve fraud detection performance.

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
📈 Model Evaluation
To properly evaluate performance on imbalanced data, the following metrics were used:

ROC-AUC Score

Confusion Matrix

Recall (Fraud Class)

🔢 Results
ROC-AUC: ~0.97

Fraud Recall: ~87%

Missed Fraud Cases (FN): 13

📊 Visualizations
All graphs were created using Matplotlib:

ROC Curve (Dark Theme)

Confusion Matrix Heatmap

Train vs Test ROC Curve

Feature Importance Bar Chart

Images were saved using:

plt.savefig("image.png", dpi=300, bbox_inches="tight")
🗂️ Project Structure

XGboost/
│
├── _confusion_matrix.png        # Confusion Matrix graph
├── _roc.png                     # ROC Curve graph
├── creditcard.csv               # Original dataset (not pushed due to size)
├── creditcard.csv.zip           # Dataset backup
├── train_Model.ipynb            # Model training notebook
├── xgboost_fraud_model.pkl      # Saved trained model
└── README.md                    # Project documentation
📊 Dataset Information
⚠️ Note:
The dataset file is very large and exceeds GitHub’s file size limit,
so it is not uploaded to this repository.

You can download the dataset from Kaggle:

🔗 Kaggle Dataset Link
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place creditcard.csv in the project folder.

▶️ How to Run the Project
1️⃣ Clone the Repository

git clone https://github.com/kanha165/Credit-Card-Fraud-Detection-using-XGBoost.git
cd Credit-Card-Fraud-Detection-using-XGBoost
2️⃣ Install Dependencies

pip install -r requirements.txt
3️⃣ Run the Notebook
Open train_Model.ipynb and run the cells step by step.

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

👤 Author
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
