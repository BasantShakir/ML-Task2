# Machine Learning Tasks - Mahmoud Hatem Elsherbiney

A collection of fundamental machine learning algorithms implemented using scikit-learn, covering regression, classification, and real-world prediction tasks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Tasks Description](#tasks-description)
  - [Task 1: Regression Models](#task-1-regression-models)
  - [Task 2: Classification Models](#task-2-classification-models)
  - [Task 3: Real-World Application](#task-3-real-world-application)
- [Installation](#installation)
- [Usage](#usage)
- [Author](#author)

---

## 🎯 Overview

This repository contains three Jupyter notebooks demonstrating the implementation of various supervised learning algorithms. Each task focuses on different machine learning techniques with practical examples.

**Covered Algorithms:**
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Pipeline with Preprocessing

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
matplotlib
```

---

## 📚 Tasks Description

### Task 1: Regression Models

**File:** `Task_1.ipynb`

#### 1.1 Linear Regression - Salary Prediction
Predicts salary based on years of experience using linear regression.

- **Dataset:** Years of experience (1-6 years) with corresponding salaries
- **Objective:** Predict salary for 7 years of experience
- **Features:** 
  - Independent variable: Years of experience
  - Dependent variable: Salary (in thousands)
- **Output:** Predicted salary and visualization of the regression line

#### 1.2 Logistic Regression - Pass/Fail Prediction
Classifies students as Pass or Fail based on study hours.

- **Dataset:** Study hours (1-7) with binary outcomes (Pass/Fail)
- **Objective:** Predict pass/fail for 4.5 study hours
- **Features:**
  - Independent variable: Study hours
  - Dependent variable: Pass (1) or Fail (0)
- **Output:** Prediction result, probability score, and decision boundary visualization

---

### Task 2: Classification Models

**File:** `Task_2.ipynb`

#### 2.1 K-Nearest Neighbors (KNN) - Student Performance
Predicts whether a student will pass or fail based on study and sleep hours.

- **Dataset:** 10 students with study hours, sleep hours, and outcomes
- **Model:** KNN with k=3 neighbors
- **Features:**
  - Study hours per day
  - Sleep hours per night
  - Target: Pass (1) or Fail (0)
- **Test Case:** Student with 6 hours study and 7 hours sleep
- **Output:** Pass/Fail prediction

#### 2.2 Naive Bayes - Tennis Play Decision
Determines whether to play tennis based on weather conditions.

- **Dataset:** 14 instances with weather attributes
- **Model:** Categorical Naive Bayes
- **Features:**
  - Outlook: Sunny, Overcast, Rainy
  - Temperature: Hot, Mild, Cool
  - Humidity: High, Normal
  - Windy: True, False
- **Test Case:** Sunny, Mild temperature, Normal humidity, Not windy
- **Output:** Yes/No decision for playing tennis

---

### Task 3: Real-World Application

**File:** `Task_3.ipynb`

#### Calories Burned Prediction Pipeline
Predicts calories burned during exercise using a complete ML pipeline with preprocessing.

- **Dataset:** CSV file containing exercise and personal metrics
- **Model:** Linear Regression with preprocessing pipeline
- **Pipeline Components:**
  1. **StandardScaler:** Normalizes numerical features
  2. **OneHotEncoder:** Encodes categorical features
  3. **LinearRegression:** Makes predictions
- **Features:**
  - Age, Weight, Height
  - Exercise duration
  - Heart rate
  - Activity type (categorical)
- **Workflow:**
  1. Load and explore data
  2. Separate numerical and categorical features
  3. Build preprocessing pipeline
  4. Train/test split (80/20)
  5. Model training and evaluation
  6. Predict calories for new input
- **Output:** RMSE score and predicted calories burned

---

## 🚀 Installation

1. **Clone or download this repository**

2. **Install required packages:**
```bash
pip install numpy pandas scikit-learn matplotlib
```

3. **For Task 3, ensure you have the dataset:**
   - Place `calories.csv` in the same directory as the notebooks

---

## 💻 Usage

1. **Open Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Navigate to the desired task:**
   - `Task_1.ipynb` - Regression models
   - `Task_2.ipynb` - Classification models
   - `Task_3.ipynb` - Calories prediction with pipeline

3. **Run all cells** to see the results and visualizations

4. **Modify the test data** in the prediction sections to experiment with different inputs

---

## 👨‍💻 Author

**Mahmoud Hatem Elsherbiney**

*Machine Learning Tasks - Level 4*

---

## 📝 Notes

- Each task includes clear comments explaining the workflow
- Visualizations are included where applicable
- The pipeline approach in Task 3 demonstrates production-ready ML practices

---



