# 💳 Credit Risk Assessment & Loan Approval Prediction

## 🎯 Project Goal

The main objective of this project is to build a machine learning model that can predict whether a customer is a **good or bad credit risk**, helping in loan approval decisions.

---

## 📂 Dataset

The dataset used is **German Credit Data**, which contains customer details such as:

* Age
* Job
* Housing
* Saving accounts
* Checking account
* Credit amount
* Duration
* Risk (Target variable)

---

## 🧮 Step-by-Step Work Done

### 🔹 1. Data Loading

```python
df = pd.read_csv("german_credit_data.csv")
```

👉 Loaded the dataset and checked its shape, structure, and data types.

---

### 🔹 2. Data Understanding

* Checked missing values using:

```python
df.isnull().sum()
```

* Viewed dataset summary:

```python
df.describe()
```

* Checked duplicates:

```python
df.duplicated().sum()
```

---

### 🔹 3. Data Cleaning

* Removed missing values:

```python
df = df.dropna().reset_index(drop=True)
```

* Dropped unnecessary column:

```python
df.drop(columns='Unnamed: 0', inplace=True)
```

👉 This step ensures clean and usable data.

---

### 🔹 4. Exploratory Data Analysis (EDA)

Performed multiple visualizations to understand patterns:

* 📊 Histogram → Distribution of Age, Credit Amount, Duration
* 🔥 Heatmap → Correlation between numerical features
* 📉 Group Analysis:

```python
df.groupby("Job")["Credit amount"].mean()
```

* 📊 Pivot Table:

```python
pd.pivot_table(df, values="Credit amount", index="Housing", columns="Purpose")
```

* 📌 Scatter Plot → Relationship between Age & Credit
* 📦 Violin Plot → Distribution of credit amount across saving accounts
* 📊 Risk Distribution:

```python
df["Risk"].value_counts(normalize=True)
```

👉 These steps helped understand data relationships and trends.

---

### 🔹 5. Feature Selection

```python
features = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration']
target = "Risk"
```

👉 Selected important columns for model training.

---

### 🔹 6. Data Encoding

Used **Label Encoding** for categorical variables:

```python
from sklearn.preprocessing import LabelEncoder
```

* Converted categorical features into numerical values
* Encoded target variable (Risk → 0/1)

👉 Saved encoders:

```python
joblib.dump(le_target, "target_encoder.joblib")
```

---

### 🔹 7. Train-Test Split

```python
train_test_split(x, y, test_size=0.2, stratify=y)
```

👉 Split data into:

* 80% training
* 20% testing

---

### 🔹 8. Model Building

n this project, I experimented with multiple machine learning models to find the best one for predicting credit risk. I also used GridSearchCV to tune hyperparameters and improve performance.

🌳 Decision Tree Classifier
Accuracy: 58.09%
Best Parameters:
max_depth = 5
min_samples_split = 2
min_samples_leaf = 1

👉 The Decision Tree model performed moderately. It was able to capture some patterns in the data, but its accuracy was relatively low, indicating limited generalization.

🌲 Random Forest Classifier
Accuracy: 61.90%
Best Parameters:
n_estimators = 100
max_depth = None
min_samples_split = 10
min_samples_leaf = 2

👉 Random Forest performed better than Decision Tree. Since it combines multiple trees, it reduced overfitting and improved overall accuracy.

🌴 Extra Trees Classifier
Accuracy: 62.85%
Best Parameters:
n_estimators = 50
max_depth = 3
min_samples_split = 2
min_samples_leaf = 1

👉 Extra Trees gave slightly better results than Random Forest. It uses more randomness, which helped in improving generalization.

⚡ XGBoost Classifier (Best Model)
Accuracy: 66.67%
Best Parameters:
n_estimators = 200
max_depth = 3
learning_rate = 0.1
colsample_bytree = 0.7
subsample = 1

👉 XGBoost performed the best among all models. It handled the data more effectively by using boosting techniques, which improved prediction accuracy.

### 🔹 9. Model Saving

Saved trained model using:

```python
joblib.dump(model, "best_model.joblib")
```

👉 Helps in deployment without retraining.

---

### 🔹 10. Deployment (Streamlit App)

Built a **Streamlit app** to:

* Take user input
* Predict credit risk in real-time

👉 Used saved model and encoders

---

## 📊 Key Insights

* Customers with higher credit amount tend to have higher risk
* Duration plays an important role in prediction
* Some categorical features like Housing and Saving accounts are significant

---

## 🚀 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib & Seaborn
* Scikit-learn
* Joblib
* Streamlit

---

## 📢 Final Conclusion

👉 The model successfully predicts whether a customer is a **good or bad credit risk**.

👉 Proper preprocessing and encoding significantly improved model performance.

👉 The deployed Streamlit app allows **real-time prediction**, making the project practical and useful.

---

## 👨‍💻 Author

**Akash Deep**
Aspiring Data Scientist

---

⭐ If you like this project, give it a star!
