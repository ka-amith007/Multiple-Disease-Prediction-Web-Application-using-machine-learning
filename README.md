# 🏥 Multiple Disease Prediction Web Application

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Machine Learning Models](#-machine-learning-models)
- [Dataset Description](#-dataset-description)
- [Installation Guide](#-installation-guide)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Future Scope](#-future-scope)
- [Authors & Credits](#-authors--credits)

---

## 🎯 Project Overview

The **Multiple Disease Prediction Web Application** is an advanced healthcare solution that leverages machine learning algorithms to predict the likelihood of multiple diseases based on user-provided medical parameters. This system aims to assist in early disease detection and provide preliminary health insights to users.

The application uses supervised learning algorithms trained on medical datasets to predict **7 different diseases**, including:
- General Disease Prediction (from symptoms)
- Diabetes
- Heart Disease
- Parkinson's Disease
- Liver Disease
- Hepatitis C
- Lung Cancer

---

## 🔍 Problem Statement

Early detection of diseases is crucial for effective treatment and improved patient outcomes. However, many individuals lack access to immediate medical consultations or are unaware of potential health risks. Traditional diagnostic processes can be:
- Time-consuming
- Expensive
- Inaccessible in remote areas
- Requiring multiple specialist consultations

This project addresses these challenges by providing an **accessible, user-friendly, and intelligent web-based platform** that can:
1. Predict multiple diseases from medical parameters
2. Provide disease descriptions and precautionary measures
3. Offer instant preliminary health assessments
4. Assist in early disease detection

---

## 🎯 Objectives

1. **Develop a Multi-Disease Prediction System**: Create a comprehensive platform capable of predicting multiple diseases using machine learning
2. **Improve Healthcare Accessibility**: Provide a web-based solution accessible to anyone with internet connectivity
3. **Early Disease Detection**: Enable users to identify potential health risks at an early stage
4. **Educational Tool**: Inform users about disease symptoms, descriptions, and precautions
5. **Data-Driven Insights**: Utilize machine learning models trained on real medical datasets for accurate predictions
6. **User-Friendly Interface**: Design an intuitive interface that simplifies complex medical assessments

---

## ✨ Features

### 🔹 **Core Features**
- **Multi-Disease Prediction**: Supports prediction of 7 different diseases
- **Symptom-Based Diagnosis**: General disease prediction from user-selected symptoms
- **Parameter-Based Predictions**: Specialized predictions using medical test parameters
- **Interactive UI**: Clean, responsive Streamlit-based web interface
- **Real-Time Predictions**: Instant disease prediction with probability scores
- **Disease Information**: Detailed descriptions and precautionary measures for predicted diseases
- **Visual Feedback**: Positive/negative result images for better user understanding

### 🔹 **Disease Prediction Modules**

| Disease | Input Parameters | Model Type |
|---------|-----------------|------------|
| **General Disease** | Symptoms selection | Scikit-learn Classifier |
| **Diabetes** | Pregnancies, Glucose, BP, Insulin, BMI, Age, etc. | SVM/Random Forest |
| **Heart Disease** | Age, Sex, Chest Pain Type, BP, Cholesterol, ECG, etc. | Logistic Regression/SVM |
| **Parkinson's** | Voice measurements (MDVP, Jitter, Shimmer, etc.) | SVM |
| **Liver Disease** | Age, Gender, Bilirubin, Proteins, Enzymes, etc. | Random Forest |
| **Hepatitis C** | Age, Gender, ALB, ALP, ALT, AST, BIL, etc. | Gradient Boosting |
| **Lung Cancer** | Gender, Age, Smoking, Anxiety, Chronic Disease, etc. | SVM/Random Forest |

---

## 🛠️ Tech Stack

### **Frontend**
- **Streamlit** - Web application framework
- **Streamlit Option Menu** - Navigation sidebar
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Data visualization
- **Pillow (PIL)** - Image processing

### **Backend & ML**
- **Python 3.x** - Core programming language
- **Scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Joblib** - Model serialization

### **Machine Learning Models**
- Support Vector Machine (SVM)
- Random Forest Classifier
- Logistic Regression
- Gradient Boosting Classifier

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
│              (Streamlit Web Application)                │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              Input Processing Layer                     │
│   • Symptom Selection                                   │
│   • Medical Parameter Input                             │
│   • Data Validation & Preprocessing                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│           Machine Learning Engine                       │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│   │ XGBoost  │  │   SVM    │  │  Random  │            │
│   │  Model   │  │  Models  │  │  Forest  │            │
│   └──────────┘  └──────────┘  └──────────┘            │
│   ┌──────────┐  ┌──────────┐                           │
│   │ Logistic │  │ Gradient │                           │
│   │Regression│  │ Boosting │                           │
│   └──────────┘  └──────────┘                           │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│            Prediction Output Layer                      │
│   • Disease Prediction                                  │
│   • Probability Score                                   │
│   • Disease Description                                 │
│   • Precautionary Measures                              │
└─────────────────────────────────────────────────────────┘
```

### **Workflow**
1. **User Input**: User selects symptoms or enters medical parameters
2. **Data Preprocessing**: Input data is validated and preprocessed
3. **Model Selection**: Appropriate ML model is selected based on disease type
4. **Prediction**: Model generates prediction with confidence score
5. **Output Display**: Results, descriptions, and precautions are displayed

---

## 🤖 Machine Learning Models

### **1. General Disease Prediction**
- **Algorithm**: Scikit-learn Classifier
- **Features**: 132 symptoms
- **Output**: Disease name with probability score
- **Accuracy**: ~95%

### **2. Diabetes Prediction**
- **Algorithm**: Support Vector Machine (SVM)
- **Features**: 8 parameters (Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age)
- **Dataset**: Pima Indians Diabetes Database
- **Accuracy**: ~77%

### **3. Heart Disease Prediction**
- **Algorithm**: Logistic Regression
- **Features**: 13 parameters (Age, Sex, Chest Pain Type, BP, Cholesterol, ECG, etc.)
- **Dataset**: UCI Heart Disease Dataset
- **Accuracy**: ~85%

### **4. Parkinson's Disease Prediction**
- **Algorithm**: Support Vector Machine (SVM)
- **Features**: 22 voice measurement parameters
- **Dataset**: UCI Parkinson's Dataset
- **Accuracy**: ~90%

### **5. Liver Disease Prediction**
- **Algorithm**: Random Forest Classifier
- **Features**: 10 parameters (Age, Gender, Bilirubin levels, Enzymes, Proteins)
- **Dataset**: Indian Liver Patient Dataset
- **Accuracy**: ~75%

### **6. Hepatitis C Prediction**
- **Algorithm**: Gradient Boosting Classifier
- **Features**: 12 parameters (Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT)
- **Dataset**: Hepatitis C Dataset
- **Accuracy**: ~80%

### **7. Lung Cancer Prediction**
- **Algorithm**: Random Forest Classifier
- **Features**: 15 parameters (Gender, Age, Smoking, Anxiety, Chronic Disease, etc.)
- **Dataset**: Survey Lung Cancer Dataset
- **Accuracy**: ~88%

---

## 📊 Dataset Description

### **Datasets Used**

| Disease | Dataset | Samples | Features | Source |
|---------|---------|---------|----------|--------|
| General Disease | Symptom Dataset | 4,920 | 132 symptoms | Kaggle |
| Diabetes | Pima Indians Dataset | 768 | 8 | UCI ML Repository |
| Heart Disease | Heart Disease Dataset | 303 | 13 | UCI ML Repository |
| Parkinson's | Parkinson's Dataset | 195 | 22 | UCI ML Repository |
| Liver Disease | Indian Liver Patient | 583 | 10 | UCI ML Repository |
| Hepatitis C | Hepatitis C Dataset | 615 | 12 | UCI ML Repository |
| Lung Cancer | Survey Lung Cancer | 309 | 15 | Kaggle |

### **Data Preprocessing**
- Handling missing values
- Feature scaling and normalization
- Encoding categorical variables
- Train-test split (80-20)
- Cross-validation for model evaluation

---

## 💻 Installation Guide

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum
- Internet connection (for first-time setup)

### **Step 1: Clone the Repository**
```bash
git clone <your-repository-url>
cd Multiple-Disease-Prediction-Web-Application-using-machine-learning-main
```

### **Step 2: Create Virtual Environment (Recommended)**
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows Command Prompt
venv\Scripts\activate.bat

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Verify Installation**
```bash
python -c "import streamlit; import sklearn; print('All packages installed successfully!')"
```

---

## 🚀 How to Run

### **Method 1: Using Streamlit Command**
```bash
streamlit run app.py
```

### **Method 2: Using Python Module**
```bash
python -m streamlit run app.py
```

### **Method 3: Specify Port**
```bash
streamlit run app.py --server.port 8080
```

### **Access the Application**
After running the command, the application will automatically open in your default browser at:
```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to the URL shown in the terminal.

---

## 📁 Project Structure

```
Multiple-Disease-Prediction-Web-Application-using-machine-learning-main/
│
├── app.py                          # Main Streamlit application
├── packages.txt                    # Streamlit package manifest
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python runtime version
│
├── 2022/                           # Disease data files
│   ├── dataset.csv
│   ├── disease_description.csv
│   ├── disease_precaution.csv
│   ├── symptom_severity.csv
│   ├── Testing.csv
│   └── Training.csv
│
├── assets/                         # UI assets
│
├── code/                           # Core application modules
│   ├── __init__.py
│   ├── DiseaseModel.py
│   ├── gen_icons.py
│   ├── helper.py
│   └── train.py
│
├── data/                           # Cleaned datasets and symptom references
│   ├── clean_dataset.tsv
│   ├── dataset.csv
│   ├── lung_cancer.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
│
├── Datasets/                       # Original datasets
│   ├── cirrhosis.csv
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── HepatitisCdata.csv
│   ├── indian_liver_patient.csv
│   ├── liver.csv
│   ├── parkinsons.csv
│   ├── survey lung cancer.csv
│   │
│   └── Code/                       # Jupyter notebooks
│       ├── diabetes disease.ipynb
│       ├── disease_prediction_model.ipynb
│       ├── Heart Disease.ipynb
│       ├── hepatitisc.ipynb
│       ├── jaundice.ipynb
│       ├── liver.ipynb
│       ├── lung_cancer.ipynb
│       └── parkinsons Disease .ipynb
│
├── model/                          # XGBoost model
│   ├── disease_model.joblib
│   └── xgboost_model.json
│
└── models/                         # Trained ML models
    ├── diabetes_model.sav
    ├── heart_disease_model.sav
    ├── hepititisc_model.sav
    ├── liver_model.sav
    ├── lung_cancer_model.sav
    └── parkinsons_model.sav
```

---

## 🔮 Future Scope

### **Short-term Enhancements**
1. **Add More Diseases**: Expand to include more disease predictions (e.g., COVID-19, Thyroid, Asthma)
2. **Mobile Application**: Develop Android/iOS mobile apps using React Native or Flutter
3. **User Authentication**: Implement user login system to save prediction history
4. **PDF Reports**: Generate downloadable PDF reports of predictions
5. **Multi-language Support**: Add support for regional languages

### **Long-term Enhancements**
1. **Deep Learning Integration**: Implement neural networks for improved accuracy
2. **Medical Image Analysis**: Add capability to analyze X-rays, CT scans, MRI images
3. **Doctor Consultation**: Integrate telemedicine for doctor consultations
4. **Wearable Device Integration**: Connect with fitness trackers and smartwatches
5. **Personalized Health Recommendations**: Provide diet and exercise recommendations
6. **Blockchain for Data Security**: Implement blockchain for secure medical records
7. **AI Chatbot**: Add conversational AI for health queries
8. **Hospital Integration**: API integration with hospital management systems

### **Research Opportunities**
- Explore ensemble learning techniques for better accuracy
- Implement explainable AI (XAI) for model interpretability
- Research federated learning for privacy-preserving model training
- Investigate transfer learning from large medical datasets

---

## 👥 Authors & Credits

### **Development Team**
- **Project Team** - Final Year Engineering Students
- **Academic Project** - Multiple Disease Prediction System

### **Academic Supervision**
- Project Guide/Supervisor
- Department of Computer Engineering

### **Acknowledgments**
- UCI Machine Learning Repository for datasets
- Kaggle community for additional datasets
- Streamlit for the amazing web framework
- Scikit-learn developers
- Stack Overflow community for technical support

### **Data Sources**
- UCI Machine Learning Repository
- Kaggle Datasets
- National Institute of Diabetes and Digestive and Kidney Diseases

### **References**
1. Priyanka Sonar, Prof. K. Jaya Malini, "DIABETES PREDICTION USING DIFFERENT MACHINE LEARNING APPROACHES", 2019 IEEE, 3rd International Conference on Computing Methodologies and Communication (ICCMC)
2. Archana Singh, Rakesh Kumar, "Heart Disease Prediction Using Machine Learning Algorithms", 2020 IEEE, International Conference on Electrical and Electronics Engineering (ICE3)
3. A. Sivasangari, Baddigam Jaya Krishna Reddy, Annamareddy Kiran, P. Ajitha, "Diagnosis of Liver Disease using Machine Learning Models", 2020 Fourth International Conference on I-SMAC (IoT in Social, Mobile, Analytics and Cloud)
4. Scikit-learn Documentation: https://scikit-learn.org/
5. Streamlit Documentation: https://docs.streamlit.io/
6. Scikit-learn Documentation: https://scikit-learn.org/

---

## 📄 License

This project is developed for educational purposes as part of a final year academic project. 

**Academic Use Only** - Not intended for commercial medical diagnosis.

---

## ⚠️ Disclaimer

**IMPORTANT**: This application is designed for educational and research purposes only. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. 

- Always seek the advice of qualified health providers with any questions regarding medical conditions
- Never disregard professional medical advice or delay seeking it because of predictions from this application
- The predictions are based on machine learning models and may not be 100% accurate
- This tool is meant to assist in preliminary health assessment only

---

## 🌟 Show Your Support

If you find this project helpful, please consider:
- ⭐ Starring the repository
- 🍴 Forking for your own experimentation
- 📢 Sharing with others
- 🐛 Reporting bugs or suggesting features

---

**Made with ❤️ for Healthcare Innovation**

*Last Updated: December 2025*

