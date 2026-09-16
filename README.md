# ❤️ Heart Disease Prediction

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">

  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">

  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">

  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">

  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">

</p>

<p align="center">
  <b>End-to-End Machine Learning Pipeline for Heart Disease Classification</b>
</p>

<p align="center">
  Data Preprocessing • EDA • Feature Engineering • Model Training • GridSearchCV • Model Evaluation
</p>

---

## 📌 Overview

**Heart Disease Prediction** is an end-to-end machine learning project designed to classify whether a patient is likely to belong to the **heart-disease** or **non-heart-disease** class based on clinical and demographic features.

The project implements a structured machine learning workflow covering:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Feature scaling
- Multiple classification algorithms
- Hyperparameter optimization using `GridSearchCV`
- Cross-validation
- Model evaluation
- Comparative model analysis

Six classical machine learning algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Support Vector Machine
5. Random Forest
6. AdaBoost

---

## 🎯 Project Objectives

The primary objectives are to:

- Build a reproducible machine learning pipeline for binary classification.
- Analyze clinical features associated with the target variable.
- Compare multiple supervised learning algorithms.
- Optimize model hyperparameters using cross-validation.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Analyze the generalization performance of the trained models.
- Provide a clean foundation for future model deployment.

---

# 🧠 Machine Learning Pipeline

```text
                    ┌─────────────────────┐
                    │      Raw Dataset    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Validation     │
                    │ & Cleaning          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Exploratory Data    │
                    │ Analysis            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ & Encoding          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Scaling     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Train / Test Split  │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │       Model Development          │
              │                                  │
              │ Logistic Regression              │
              │ Decision Tree                    │
              │ KNN                              │
              │ SVM                              │
              │ Random Forest                    │
              │ AdaBoost                         │
              └───────────────┬──────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   GridSearchCV      │
                    │ Hyperparameter      │
                    │ Optimization        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Cross Validation    │
                    │ & Evaluation        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Comparison    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Final Prediction    │
                    └─────────────────────┘


📊 Dataset

The dataset contains clinical and demographic attributes used for binary heart disease classification.

Feature Description
Feature	Description
age	Age of the patient
sex	Sex of the patient
cp	Chest pain type
trestbps	Resting blood pressure
chol	Serum cholesterol
fbs	Fasting blood sugar
restecg	Resting electrocardiographic results
thalach	Maximum heart rate achieved
exang	Exercise-induced angina
oldpeak	ST depression induced by exercise
slope	Slope of the peak exercise ST segment
ca	Number of major vessels
thal	Thalassemia-related measurement
target	Binary target variable
Target Variable
0 → No Heart Disease
1 → Heart Disease
🧹 Data Preprocessing

The dataset was processed before training the machine learning models.

The preprocessing workflow includes:

Loading the dataset.
Inspecting dataset structure.
Checking data types.
Checking missing values.
Checking duplicate records.
Analyzing numerical features.
Analyzing categorical features.
Encoding categorical variables where required.
Scaling numerical features where required.
Separating independent features and target variable.
Splitting the data into training and testing sets.

Example:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the underlying structure and relationships within the dataset.

EDA Areas
Target class distribution
Age distribution
Gender distribution
Chest pain distribution
Cholesterol distribution
Resting blood pressure
Maximum heart rate
Exercise-induced angina
Correlation between numerical features
Feature-target relationships
Outlier analysis
Visualizations
✓ Count Plots
✓ Distribution Plots
✓ Box Plots
✓ Correlation Heatmap
✓ Feature Comparison Plots
✓ Target Distribution

EDA helps understand the data before model development and can reveal potential data quality issues and useful feature relationships.

🤖 Machine Learning Models

The following supervised classification algorithms were implemented:

Model	Type
Logistic Regression	Linear Classification
Decision Tree	Tree-Based Classification
KNN	Distance-Based Classification
SVM	Kernel-Based Classification
Random Forest	Ensemble Learning
AdaBoost	Boosting Ensemble
⚙️ Hyperparameter Optimization

Hyperparameter tuning was performed using GridSearchCV.

GridSearchCV evaluates multiple combinations of hyperparameters using cross-validation and selects the configuration with the highest validation score according to the chosen scoring metric.

🔹 Logistic Regression
Best Parameters
C = 0.1
solver = lbfgs
Performance
Mean CV Accuracy = 0.7828
Test Accuracy    = 0.7905
F1-Score         = 0.7982
Classification Report
Class	Precision	Recall	F1-Score	Support
0	0.81	0.76	0.78	104
1	0.78	0.82	0.80	106
Accuracy			0.79	210
Macro Avg	0.79	0.79	0.79	210
Weighted Avg	0.79	0.79	0.79	210
🔹 Decision Tree
Best Parameters
criterion = entropy
max_depth = 3
Performance
Mean CV Accuracy = 0.7733
Test Accuracy    = 0.7762
F1-Score         = 0.7873
Classification Report
Class	Precision	Recall	F1-Score	Support
0	0.80	0.73	0.76	104
1	0.76	0.82	0.79	106
Accuracy			0.78	210
Macro Avg	0.78	0.78	0.78	210
Weighted Avg	0.78	0.78	0.78	210
🔹 K-Nearest Neighbors (KNN)
Best Parameters
n_neighbors = 9
weights = distance
Performance
Mean CV Accuracy = 0.7792
Test Accuracy    = 0.7619
F1-Score         = 0.7748
Classification Report
Class	Precision	Recall	F1-Score	Support
0	0.79	0.71	0.75	104
1	0.74	0.81	0.77	106
Accuracy			0.76	210
Macro Avg	0.76	0.76	0.76	210
Weighted Avg	0.76	0.76	0.76	210
🔹 Support Vector Machine (SVM)
Best Parameters
C = 0.1
kernel = rbf
Performance
Mean CV Accuracy = 0.7888
Test Accuracy    = 0.7762
F1-Score         = 0.7854
Classification Report
Class	Precision	Recall	F1-Score	Support
0	0.79	0.74	0.77	104
1	0.76	0.81	0.79	106
Accuracy			0.78	210
Macro Avg	0.78	0.78	0.78	210
Weighted Avg	0.78	0.78	0.78	210
🔹 Random Forest
Best Parameters
max_depth = 8
n_estimators = 200
Performance
Mean CV Accuracy = 0.7971
Test Accuracy    = 0.7952
F1-Score         = 0.8054
Classification Report
Class	Precision	Recall	F1-Score	Support
0	0.82	0.75	0.78	104
1	0.77	0.84	0.81	106
Accuracy			0.80	210
Macro Avg	0.80	0.79	0.79	210
Weighted Avg	0.80	0.80	0.79	210
🔹 AdaBoost
Best Parameters
learning_rate = 0.01
n_estimators = 50
Performance
Mean CV Accuracy = 0.7876
Test Accuracy    = 0.7762
F1-Score         = 0.7854
Classification Report
Class	Precision	Recall	F1-Score	Support
0	0.79	0.74	0.77	104
1	0.76	0.81	0.79	106
Accuracy			0.78	210
Macro Avg	0.78	0.78	0.78	210
Weighted Avg	0.78	0.78	0.78	210
📊 Model Comparison After GridSearchCV
Model	Test Accuracy	F1-Score	CV Accuracy
Random Forest	79.52%	80.54%	79.71%
Logistic Regression	79.05%	79.82%	78.28%
Decision Tree	77.62%	78.73%	77.33%
SVM	77.62%	78.54%	78.88%
AdaBoost	77.62%	78.54%	78.76%
KNN	76.19%	77.48%	77.92%
📐 Evaluation Metrics
Accuracy

Accuracy measures the proportion of correctly classified observations.

$$ Accuracy = \frac{TP + TN} {TP + TN + FP + FN} $$
Precision

Precision measures the proportion of predicted positive observations that are actually positive.

$$ Precision = \frac{TP} {TP + FP} $$
Recall

Recall measures the proportion of actual positive observations that are correctly identified.

$$ Recall = \frac{TP} {TP + FN} $$
F1-Score

F1-score is the harmonic mean of Precision and Recall.

$$ F1 = 2 \times \frac{Precision \times Recall} {Precision + Recall} $$
Confusion Matrix
                    Predicted
                 0           1
              ┌──────────┬──────────┐
Actual     0  │    TN    │    FP    │
              ├──────────┼──────────┤
           1  │    FN    │    TP    │
              └──────────┴──────────┘

Where:

TP = True Positive
TN = True Negative
FP = False Positive
FN = False Negative
🔍 Experimental Results

The reported experiment evaluated six classification algorithms after hyperparameter tuning.

Models Evaluated : 6
Tuning Method    : GridSearchCV

Test Accuracy Range : 76.19% – 79.52%
F1-Score Range      : 77.48% – 80.54%
CV Accuracy Range   : 77.33% – 79.71%

The reported Random Forest configuration produced:

Test Accuracy : 79.52%
F1-Score      : 80.54%
CV Accuracy   : 79.71%

For class 1, the reported Random Forest metrics were:

Precision = 0.77
Recall    = 0.84
F1-Score  = 0.81

The reported recall of 0.84 means that 84% of the actual class-1 observations in this test set were correctly identified.

These results correspond specifically to the dataset, preprocessing pipeline, train-test split, and experimental configuration used in this project.

🏗️ Project Structure
Heart-Disease-Prediction/
│
├── data/
│   └── heart.csv
│
├── notebooks/
│   └── heart_disease_prediction.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── .gitkeep
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml
🛠️ Tech Stack
Category	Technology
Programming Language	Python
Data Manipulation	Pandas
Numerical Computing	NumPy
Data Visualization	Matplotlib
Statistical Visualization	Seaborn
Machine Learning	Scikit-learn
Hyperparameter Tuning	GridSearchCV
Scientific Computing	SciPy
Development	Jupyter Notebook / VS Code
Version Control	Git
Repository Hosting	GitHub
📦 Installation
1. Clone the Repository
git clone https://github.com/<your-username>/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
2. Create a Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
📋 Requirements

Create a requirements.txt file:

numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
jupyter
notebook

For fully reproducible experiments, package versions should be pinned to the versions used during model development.

Example:

numpy==<version>
pandas==<version>
matplotlib==<version>
seaborn==<version>
scikit-learn==<version>
scipy==<version>
jupyter==<version>
notebook==<version>
▶️ Running the Project
Jupyter Notebook

Start Jupyter:

jupyter notebook

Open:

notebooks/heart_disease_prediction.ipynb

Run the notebook cells sequentially.

VS Code

Open the repository:

code .

Select the project's Python virtual environment and execute the notebook or Python scripts.

🧪 Example Prediction

After training the model, a new patient observation can be passed to the trained classifier.

prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("Heart disease predicted")
else:
    print("No heart disease predicted")

For probability estimates:

probability = model.predict_proba(new_patient)

print(probability)

The prediction is a machine-learning output and should not be interpreted as a medical diagnosis.

🔁 Reproducibility

The project follows reproducibility practices including:

Fixed random seeds
Explicit train-test split
Stratified sampling where applicable
Explicit hyperparameter configuration
Cross-validation
Documented evaluation metrics

Example:

train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

For complete reproducibility, document:

Python Version
Dataset Version
Library Versions
Random Seed
Train/Test Split
Preprocessing Pipeline
Hyperparameter Search Space
Cross-Validation Strategy
Evaluation Metrics
🧠 Explainable AI

Healthcare-oriented machine learning systems can benefit from explainability.

Future versions can integrate:

SHAP

SHAP can be used to analyze:

Global feature importance
Feature contribution
Individual prediction explanations
Trained Model
      ↓
    SHAP
      ↓
Feature Contributions
      ↓
Model Explanation
LIME

LIME can provide local explanations for individual model predictions by approximating model behavior around a specific observation.

🌐 Deployment Architecture

A possible production-oriented architecture is:

                         ┌───────────────┐
                         │     User      │
                         └───────┬───────┘
                                 │
                                 ▼
                     ┌─────────────────────┐
                     │ Streamlit / Web UI  │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   FastAPI Backend   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Preprocessing       │
                     │ Pipeline            │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Trained ML Model    │
                     │ Random Forest       │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Prediction +        │
                     │ Probability         │
                     └─────────────────────┘
🚀 Future Improvements

Potential improvements include:

 ROC-AUC evaluation
 Precision-Recall curves
 Stratified K-Fold cross-validation
 Feature selection
 Automated feature engineering
 Bayesian hyperparameter optimization
 Model calibration
 SHAP explainability
 LIME explainability
 MLflow experiment tracking
 Streamlit deployment
 FastAPI REST API
 Docker containerization
 GitHub Actions CI/CD
 External validation dataset
 Model monitoring
 Data drift detection
 Prediction logging
🔐 Responsible AI & Limitations

This project is intended for educational and research purposes.

The model should not be treated as an independent clinical decision-making system.

Model performance can be affected by:

Dataset size
Dataset quality
Sampling strategy
Missing information
Data distribution
Population differences
Feature engineering
Preprocessing choices
Model assumptions

The reported metrics should not be interpreted as evidence of clinical effectiveness.

Clinical diagnosis and treatment decisions should be made by qualified healthcare professionals using appropriate medical evidence.

📜 Licensing
Project License

This project is released under the MIT License.

MIT License

Copyright (c) 2026 Parul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
📚 Third-Party Libraries & Licenses

This project uses open-source Python libraries maintained by their respective communities.

Library	Purpose	License
Python	Programming Language	Python Software Foundation License
NumPy	Numerical Computing	BSD 3-Clause
Pandas	Data Manipulation	BSD 3-Clause
Matplotlib	Data Visualization	Matplotlib License
Seaborn	Statistical Visualization	BSD 3-Clause
Scikit-learn	Machine Learning	BSD 3-Clause
SciPy	Scientific Computing	BSD 3-Clause
Jupyter	Interactive Computing	BSD 3-Clause
Notebook	Jupyter Notebook Environment	BSD 3-Clause

Third-party libraries remain the property of their respective copyright holders. Their licenses apply independently of this project's MIT License.

📄 License Files

For a professional repository, include the following file:

LICENSE

The LICENSE file should contain the project's MIT License text.

If third-party source code is copied, modified, or redistributed, retain the applicable copyright and license notices required by the respective projects.

📌 Results Summary
Models Evaluated : 6
Tuning Method    : GridSearchCV

Test Accuracy Range : 76.19% – 79.52%
F1-Score Range      : 77.48% – 80.54%
CV Accuracy Range   : 77.33% – 79.71%
Complete Results
Model	Test Accuracy	F1-Score	CV Accuracy
Random Forest	79.52%	80.54%	79.71%
Logistic Regression	79.05%	79.82%	78.28%
Decision Tree	77.62%	78.73%	77.33%
SVM	77.62%	78.54%	78.88%
AdaBoost	77.62%	78.54%	78.76%
KNN	76.19%	77.48%	77.92%
📎 Project Deliverables
├── Dataset
├── Exploratory Data Analysis
├── Data Preprocessing
├── Feature Engineering
├── Six ML Classification Models
├── GridSearchCV Hyperparameter Tuning
├── Cross-Validation
├── Classification Reports
├── Model Comparison
└── Prediction Pipeline
🧾 Citation

If you use this repository for academic work, research, or further development, please provide appropriate attribution to the original dataset and the open-source libraries used.

🤝 Contributing

Contributions are welcome.

Create a feature branch
git checkout -b feature/your-feature
Stage changes
git add .
Commit changes
git commit -m "Add: your feature"
Push the branch
git push origin feature/your-feature

Then open a Pull Request.

🐛 Issues

If you find a bug, data issue, reproducibility problem, or improvement opportunity, please open a GitHub Issue with:

1. Problem description
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Python version
6. Library versions
7. Relevant error message
👩‍💻 Author

Parul

Postgraduate Student | Artificial Intelligence & Machine Learning

⭐ Acknowledgements

This project makes use of the Python scientific computing and machine learning ecosystem, including:

NumPy
Pandas
Matplotlib
Seaborn
SciPy
Scikit-learn
Jupyter

All third-party libraries remain under their respective open-source licenses.

<p align="center"> <b>Built with Python • Machine Learning • Data Science</b> </p> <p align="center"> ⭐ Star this repository if you find it useful. </p> ```
