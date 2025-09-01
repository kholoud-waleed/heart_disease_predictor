# 🩺❤️ Heart Disease Prediction – Machine Learning Full Pipeline  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)  
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy) 
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=plotly)  
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-0e76a8)  
![Streamlit](https://img.shields.io/badge/Streamlit-App%20Deployment-ff4b4b?logo=streamlit)  

---

This repository contains a **complete end-to-end Machine Learning pipeline** built on the **Heart Disease UCI Dataset**.  
The project demonstrates the full ML workflow, that involves **data preprocessing**, **feature selection**, **dimensionality reduction (PCA)**, **model training & evaluation**, and **deployment** using a **Streamlit web app**. 

---

## 📂 Project Structure: 

### 01 - Data Preprocessing  
- Loaded the Heart Disease UCI dataset into a Pandas DataFrame.
- Handled missing values (imputation or removal).
- Performed one-hot encoding for categorical variables.
- Standardized numerical features using StandardScaler.
- Conduct Exploratory Data Analysis (EDA) with histograms, correlation heatmaps and boxplots.

### 02 - PCA Analysis  
- Applied PCA to reduce feature dimensionality while maintaining variance.
- Determined the optimal number of principal components using the explained variance ratio.
- isualized PCA results using a scatter plot and cumulative variance plot.

### 03 - Feature Selection  
- Used Feature Importance (Random Forest / XGBoost feature importance scores) to rank variables.
- Applied Recursive Feature Elimination (RFE) to select the best predictors.
- Used Chi-Square Test to check feature significance.
- Selected only the most relevant features for modeling.

### 04 - Supervised Learning  - Classification
- Splitted the dataset into training (80%) and testing (20%) sets.
- Trained the following models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Evaluated models using:
  - Accuracy, Precision, Recall, F1-score
  - ROC Curve & AUC Score

### 05 - Unsupervised Learning  - Clustering
- Applied clustering algorithms for pattern discovery:  
  - K-Means Clustering  
  - Hierarchical Clustering  
- Compared clusters with actual disease labels.

### 06 - Model Optimization & Export  
- Performed **Hyperparameter Tuning** using:  
  - GridSearchCV  
  - RandomizedSearchCV
- Compared optimized models with baseline performance.
- Exported best-performing model as `svm_model.pkl` and saved evaluation results in `model_results.txt`.  

### 07 - Deployment using Streamlit UI App  
- Created a Streamlit UI `app.py` to allow users to input health data.
- Integrated the final trained model to provide real-time prediction output based on user inputs.
- Added data visualization for users to explore heart disease trends like Histograms and other plots.
- Built a simple **Streamlit UI** for interactive predictions.  

---

## 🚀🤖 Installation & Usage  

1. Clone this repository:  
   ```bash
   git clone https://github.com/kholoud-waleed/heart_disease_predictor.git
   cd heart_disease_predictor

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run Streamlit UI app:
   ```bash
   streamlit run ui/app.py

---

## ⚙️🛠️ Tools & Libraries Used:
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Dimensionality Reduction & Feature Selection**: PCA, Recursive Feature Elimination (RFE), Chi-Square Test
- **Supervised Learning Models**: Logistic Regression, Decision Trees, Random Forest, Support Vector Machine (SVM)
- **Unsupervised Learning Models**: K-Means Clustering, Hierarchical Clustering
- **Model Optimization**: GridSearchCV, RandomizedSearchCV
- **Deployment Tools**: Streamlit [Bonus]

---

## 📑 Dataset  & Target of this Project:

- **Dataset**: Heart Disease UCI Dataset  (https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Target**: Predict presence of heart disease (Binary Classification).  

---

## 📌 Results: 

- Compared both supervised and unsupervised models on **Accuracy, Precision, Recall, F1-score and ROC/AUC score**.  
- Achieved best performance with **SVM**.  
- Final trained model deployed in an **User-Interactive Streamlit Web App**.  

---

## 🙌 Acknowledgments: 

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) for dataset.  
- Open-source community for ML libraries.  

---

## 📝 Author:
  ### Kholoud Waleed Ali
