# AI Job Salary Predictor

A Streamlit web application that predicts the estimated salary (in USD) for AI-related job profiles based on various job features.

---

## ❤️ Credits

Made by Ankit. [Instagram](https://www.instagram.com/__ankit._.op_/)

---

## 🚀 Overview

This app loads a trained machine learning model, takes user inputs for job attributes, encodes them, and predicts salary. It includes:

* Clean UI with Streamlit
* Automatic encoding of categorical features
* Input validation and feature statistics
* User-friendly salary breakdown (annual, monthly, weekly, daily, hourly)

---

## 📂 Project Structure

```
app.py
ai_job_dataset.csv
salary_predictor_model.pkl
```

* **app.py:** Main Streamlit application
* **ai_job_dataset.csv:** Dataset used to create encoders
* **salary_predictor_model.pkl:** Pre-trained ML model used for prediction

---

## 🛠 Requirements

Install dependencies:

```bash
pip install streamlit pandas scikit-learn joblib
```

Yah:

```bash
pip install requirements.txt
```

---

## ▶️ How to Run

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 📊 Input Features

Inputs are auto-generated from the dataset. The app handles two types:

* **Categorical:** Encoded with LabelEncoder
* **Numerical:** Uses min, max, and median values for safe input ranges

---

## 🧠 Model

* Loads `salary_predictor_model.pkl`
* Expects feature order identical to dataset
* Predicts salary and displays:

  * Annual salary
  * Monthly salary
  * Weekly salary
  * Daily salary
  * Hourly salary

---

## ⚠️ Error Handling

* Missing dataset or model → visible error message
* Invalid category → clear message with incorrect value
* Prediction error → expandable error details

---

## 📁 Required Files

Make sure these exist in the same folder:

* `ai_job_dataset.csv`
* `salary_predictor_model.pkl`

---
