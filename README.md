# Employee Attrition Prediction & Credit Card Customer Segmentation  
*A repository showcasing machine learning workflows for classification and unsupervised learning tasks.*  

---

## 📁 Repository Overview  
This repository contains two distinct machine learning projects:  
1. **Employee Attrition Prediction** using KNN and ensemble classifiers.  
2. **Credit Card Customer Segmentation** via PCA and K-Means clustering.  

Both projects emphasize exploratory data analysis (EDA), preprocessing, model implementation, and performance evaluation.  

---

## � 1. Employee Attrition Prediction using KNN and Ensemble Methods  

### 📝 Project Overview  
This project predicts employee attrition (employees leaving a company) using classification algorithms. It explores K-Nearest Neighbors (KNN) and ensemble methods (Random Forest, Bagging, AdaBoost) to identify patterns in employee behavior and predict attrition risk.  

### 🔑 Key Steps  
- **Dataset Exploration & Preprocessing**:  
  - Analyzed feature distributions, encoded categorical variables, and scaled numerical features.  
  - Addressed missing data and irrelevant columns.  
- **Model Implementation**:  
  - **KNN Classifier**: Optimized hyperparameters (`n_neighbors`, `weights`) using `GridSearchCV`.  
  - **Ensemble Methods**:  
    - **Random Forest**: Leveraged feature importance analysis to identify key attrition drivers.  
    - **Bagging Classifier**: Reduced overfitting by aggregating predictions from base estimators.  
    - **AdaBoost**: Boosted weak learners iteratively to improve accuracy.  
- **Evaluation**:  
  - Compared models using accuracy, precision, recall, F1-score, and confusion matrices.  
  - Visualized feature importance to interpret model decisions.  

### 🛠️ Libraries & Tools  
- **Python**, **Pandas**, **NumPy**  
- **Scikit-learn**: `KNeighborsClassifier`, `RandomForestClassifier`, `BaggingClassifier`, `AdaBoostClassifier`, `GridSearchCV`  
- **Matplotlib**, **Seaborn** for visualizations  

---

## 💳 2. Customer Segmentation with PCA and K-Means Clustering  

### 📝 Project Overview  
This project segments credit card customers into distinct groups using unsupervised learning. It combines Principal Component Analysis (PCA) for dimensionality reduction and K-Means clustering to identify behavioral patterns.  

### 🔑 Key Steps  
- **Data Preprocessing**:  
  - Removed irrelevant columns (e.g., `CUST_ID`).  
  - Handled missing values via median imputation and addressed multicollinearity.  
- **PCA Implementation (from scratch)**:  
  - Computed covariance matrices, eigenvalues, and eigenvectors.  
  - Reduced dimensions while retaining ~85% variance.  
- **Clustering**:  
  - **K-Means (from scratch)**: Iteratively assigned customers to clusters based on PCA-transformed features.  
  - **Hierarchical Clustering**: Compared results using dendrograms and linkage matrices.  
- **Evaluation**:  
  - Measured cluster quality with silhouette scores before/after PCA.  
  - Visualized clusters in 2D/3D space to validate segmentation.  

### 🛠️ Libraries & Tools  
- **Python**, **Pandas**, **NumPy**  
- **Matplotlib**, **Seaborn** for plots  
- **SciPy** (for hierarchical clustering)  

---

## 🚀 Key Takeaways  
- **Employee Attrition Prediction**:  
  - Ensemble methods (e.g., Random Forest) outperformed KNN in accuracy and robustness.  
  - Feature importance analysis highlighted "Job Satisfaction" and "Monthly Income" as critical attrition predictors.  
- **Customer Segmentation**:  
  - PCA improved clustering efficiency by reducing noise and redundancy.  
  - K-Means identified 4 distinct customer segments with unique credit usage behaviors.  

