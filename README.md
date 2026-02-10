# 🚢 Titanic Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Machine Learning project predicting Titanic passenger survival using **Random Forest** and **Decision Tree** classifiers.

---

## Project Overview

This project analyzes the famous Titanic dataset to predict passenger survival with **83%+ accuracy**.

### Key Highlights:
- ✅ Comprehensive **Exploratory Data Analysis (EDA)**
- ✅ Advanced **Feature Engineering** (FamilySize, Title extraction, Age groups)
- ✅ **Two ML Models** compared (Random Forest vs Decision Tree)
- ✅ **Multiple Evaluation Metrics** (Accuracy, Precision, Recall, ROC-AUC)
- ✅ **Beautiful Visualizations** (Confusion matrices, Feature importance, ROC curves)

---

## Results

| Model | Training Accuracy | Testing Accuracy | ROC-AUC |
|-------|------------------|------------------|---------|
| **Random Forest** | 85.23% | **83.24%** | 0.8745 |
| **Decision Tree** | 82.58% | 79.33% | 0.8321 |

**Winner**: Random Forest 🏆

---

## 🔍 Key Insights

### Top 5 Predictors of Survival:
1. **Gender** (35% importance) - Women had 4x higher survival rate
2. **Fare** (19% importance) - Higher ticket prices = better survival
3. **Age** (14% importance) - Children prioritized in evacuation
4. **Passenger Class** (13% importance) - First-class had better access to lifeboats
5. **Family Size** (8% importance) - Medium families survived more

### Survival Statistics:
- 📊 Overall survival rate: **38.4%**
- 👩 Female survival rate: **74.2%**
- 👨 Male survival rate: **18.9%**
- 🎫 1st Class survival: **62.9%**
- 🎫 3rd Class survival: **24.2%**

---

## 📁 Project Structure
```
titanic-ml-project/
│
├── Titanic_ML_Project.ipynb    # Complete analysis notebook
│
├── models/                      # Trained models
│   ├── titanic_rf_model.pkl
│   ├── titanic_dt_model.pkl
│   └── scaler.pkl
│
├── visualizations/              # Generated plots
│   ├── survival_analysis.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── roc_curves.png
│
├── reports/
│   └── project_summary.txt
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development

---

## Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/ToobaFazal02/titanic-ml-project.git
cd titanic-ml-project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the notebook**
```bash
jupyter notebook Titanic_ML_Project.ipynb
```

---

## Model Pipeline

1. **Data Loading** - Load Titanic dataset from Kaggle
2. **EDA** - Explore patterns, missing values, distributions
3. **Preprocessing** - Handle missing values, feature engineering
4. **Feature Engineering** - Create FamilySize, IsAlone, Title, Age/Fare groups
5. **Encoding** - Convert categorical variables to numerical
6. **Scaling** - Standardize features
7. **Model Training** - Train Random Forest & Decision Tree
8. **Evaluation** - Compare models using multiple metrics
9. **Visualization** - Generate insightful plots

---

## 📸 Visualizations

### Survival Analysis
Comprehensive analysis of survival patterns by gender, class, and age.

### Confusion Matrices
Side-by-side comparison of Random Forest and Decision Tree predictions.

### Feature Importance
Top 15 features ranked by their impact on survival prediction.

### ROC Curves
Performance comparison using Receiver Operating Characteristic curves.

---

## 🎓 What I Learned

- **Data Preprocessing**: Handling missing values, outliers, and encoding
- **Feature Engineering**: Creating meaningful features from existing data
- **ML Algorithms**: Understanding Random Forest vs Decision Tree
- **Model Evaluation**: Using multiple metrics for comprehensive assessment
- **Data Visualization**: Creating professional, insightful plots
- **Best Practices**: Code organization, documentation, version control

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Try ensemble methods (XGBoost, LightGBM)
- [ ] Implement deep learning models
- [ ] Create interactive dashboard with Streamlit
- [ ] Deploy model as REST API
- [ ] Add explainability with SHAP values

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


## Acknowledgments

- Dataset from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- Inspired by the data science community
- Built with ❤️ and Python

---

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- [Random Forest Algorithm](https://en.wikipedia.org/wiki/Random_forest)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

