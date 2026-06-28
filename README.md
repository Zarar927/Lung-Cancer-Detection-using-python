# 🫁 Lung Cancer Risk Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting **lung cancer risk levels (Low, Medium, High)** using machine learning techniques based on patient health records and lifestyle factors. The dataset contains **1000 patient records** with **24 medical and environmental features**.

The goal is to analyze risk factors such as smoking, air pollution, genetic risk, and symptoms, and build models that can accurately classify lung cancer severity levels.

---

## 🎯 Objectives

* Perform data analysis on lung cancer dataset
* Check and handle missing values
* Visualize class distribution
* Train machine learning models for classification
* Compare performance of multiple algorithms
* Predict lung cancer risk level (Low / Medium / High)

---

## 📂 Dataset Information

* Total Records: **1000**
* Total Features: **26 columns**
* Target Variable: **Level**
* Classes:

  * 🟢 Low
  * 🟡 Medium
  * 🔴 High

### 📊 Features include:

* Age, Gender
* Air Pollution
* Smoking & Passive Smoking
* Alcohol Use
* Genetic Risk
* Chronic Lung Disease
* Obesity, Fatigue
* Chest Pain, Coughing of Blood
* Shortness of Breath, Wheezing
* And many more medical symptoms

---

## 🧹 Data Preprocessing

### ✅ Missing Values Check

No missing values were found in the dataset:

```python
pf.isnull().sum()
```

✔ Result: **All columns contain 0 missing values**

---

## 📊 Exploratory Data Analysis (EDA)

### 🔹 Dataset Overview

```python
pf.info()
```

* 1000 entries
* 24 numerical features
* 2 categorical features (Patient Id, Level)

---

## 🥧 Target Variable Distribution

A pie chart was used to visualize the distribution of lung cancer levels.

### 📈 Class Distribution Graph

📌 **Observation:**

* Dataset is fairly balanced
* All three classes have similar representation
* This helps improve model fairness and reduces bias

---

## 🧠 Machine Learning Models Used

Three classification algorithms were used:

### 1️⃣ Support Vector Machine (SVM)

```python
S = svm.SVC()
```

* Best performing model
* Achieved **100% testing accuracy**
* Excellent decision boundary separation

---

### 2️⃣ K-Nearest Neighbors (KNN)

```python
knn = KNeighborsClassifier()
```

* Achieved **100% training accuracy**
* Sensitive to dataset structure
* May require testing validation to confirm generalization

---

### 3️⃣ Bernoulli Naive Bayes

```python
B = BernoulliNB()
```

* Achieved **89% testing accuracy**
* Performs well but assumes feature independence
* Slightly lower performance compared to SVM

---

## 📊 Model Performance Comparison

| Model       | Accuracy         | Result                 |
| ----------- | ---------------- | ---------------------- |
| SVM         | **100% (Test)**  | 🏆 Best                |
| KNN         | **100% (Train)** | ⚠ Possible overfitting |
| Naive Bayes | **89% (Test)**   | Good baseline          |

---

## 📈 Accuracy Comparison Graph

---

## 🔍 Key Insights

* SVM performed the best with highest accuracy
* KNN perfectly fitted training data but needs proper testing validation
* Naive Bayes gave stable but lower performance
* Dataset is clean with no missing values
* Features are strongly related to lifestyle and medical conditions

---

## ⚠️ Important Note

Achieving **100% accuracy** in real-world medical datasets is uncommon. It may indicate:

* Possible overfitting
* Data leakage
* Highly structured dataset
* Duplicate or correlated samples

👉 For better evaluation, consider:

* Confusion Matrix
* Precision, Recall, F1-score
* Cross-validation (K-Fold)
* ROC-AUC Score

---

## 🚀 Future Improvements

* Apply feature scaling (StandardScaler)
* Perform hyperparameter tuning (GridSearchCV)
* Use ensemble models (Random Forest, XGBoost)
* Apply cross-validation
* Build a web app using Streamlit or Flask
* Deploy model for real-time prediction

---

## 📌 Conclusion

This project successfully demonstrates the use of machine learning techniques for lung cancer risk prediction. The Support Vector Machine (SVM) achieved the highest performance with **100% accuracy**, making it the most suitable model for this dataset.

The system can help in early identification of lung cancer risk levels based on patient health conditions and lifestyle factors, supporting medical decision-making.

---

## 👨‍💻 Technologies Used

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn (SVM, KNN, Naive Bayes)

---

## 📁 Project Structure

```
Lung-Cancer-Prediction/
│
├── dataset.csv
├── lung_cancer_model.ipynb
├── README.md
└── plots/
    ├── pie_chart.png
    └── accuracy_graph.png
```

---

## ⭐ If you like this project

Give a ⭐ on GitHub and feel free to contribute!
