 ## 🏔️ Predictive Pipeline for Genetic Analysis of Acute Mountain Sickness using K Means Clustering and Naive Bayes 🏔️ 

 > **A computational biology and machine learning project investigating genetic, physiological and transcriptomic factors associated with susceptibility to Acute Mountain Sickness (AMS).**


 # About the Project

 **Acute Mountain Sickness (AMS)** is an altitude-related condition that may occur when an individual ascends to high-altitude environments where oxygen availability is reduced.

 Although environmental and physiological factors play an important role in AMS susceptibility, genetic variations significantly contribute to differences in individual responses to hypoxic environments.

 This project investigates the relationship between **genetic, physiological and transcriptomic characteristics** to establish their relation with AMS susceptibility using computational analysis and machine learning and focuses on identifying highly dominating genetic features that may be associated with different physiological response patterns.

 The overall analysis combines:

 - 🧬 Genetic data analysis
- 🫁 Physiological response analysis
- 📊 Exploratory data analysis
- 🤖 Unsupervised machine learning : K Means Algorithm
- 🧠 Supervised machine learning : Naive Bayes Algorithm
- 🔬 Candidate gene identification
- 📈 Biological interpretation


 # Methodology

 The project development took place in the following stages:

```
Data Collection
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis
      ↓
K-Means Clustering (K=4)
      ↓
Candidate Gene Selection
      ↓
Naive Bayes Classification
      ↓
AMS Susceptibility Prediction
      ↓
Biological Interpretation
```

---

 # Clusters formed during K Means Clsutering 

The data were initially organized into the required format using spreadsheet-based preprocessing and subsequently prepared for computational analysis. **Four** distinct clusters were formed to facilitate the systematic characterization of AMS patterns, individual physiological responses and susceptibility profiles for efficient downstream analysis and detection.

![Susceptibility Profile Clusters](image.png)

 After clustering, most dominating genes associated with the identified profiles were analysed to determine potentially relevant features which were used for predicting AMS susceptibility for a person based on his/her genetic profile.


 # Tech Stack Used

- **Python 3.9+** - Core programming language for data processing, machine learning, statistical analysis and model development
- **Pandas** - Data loading, integration, cleaning, preprocessing
- **NumPy** - Numerical computation, array operations, synthetic data generation
- **SciPy** - Statistical analysis using Welch’s t-test for cluster-wise gene selection
- **Scikit-learn** - Machine learning pipeline including K-Means clustering, Gaussian Naive Bayes, feature scaling, PCA, train-test splitting and model evaluation
- **Matplotlib** - Visualization of clusters, confusion matrices and ROC curves
- **OpenPyXL** - Reading and processing Excel-based genetic and physiological datasets
- **Joblib** - Serialization and persistence of trained machine learning models and preprocessing components
- **K-Means Clustering** - Unsupervised grouping of individuals into four AMS susceptibility and acclimatization profiles
- **Gaussian Naive Bayes** - Probabilistic classification of individuals into AMS and non-AMS categories
- **PCA** - Dimensionality reduction for visualizing high-dimensional cluster patterns
- **Welch’s t-test** - Statistical identification and ranking of genes associated with individual clusters
- **Git & GitHub** - Version control and repository management for the project

### Programming and Development

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,git,github,vscode&theme=dark&perline=4" />
</p>

### Machine Learning & Data Science

<p align="center">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-0C55A5?style=for-the-badge&logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" />
</p>

### Machine Learning Methods

<p align="center">
  <img src="https://img.shields.io/badge/K--Means_Clustering-6C63FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gaussian_Naive_Bayes-00A67E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PCA-FF6B6B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Welch's_t--test-4C78A8?style=for-the-badge" />
</p>

# Results 
Successful Generation and Integration of Clinical, Physiological, Transcriptomic and Lake Louise Score Data for AMS Risk Analysis, K-Means Clustering and Naive Bayes Prediction
<img width="387" height="215" alt="Initial Stage" src="https://github.com/user-attachments/assets/7ce7567d-cde8-4c52-aed4-3559aa74d051" />
<br>
**Stage 1 of  Machine Learning-Based Genetic Analysis of AMS pipeline**  : Data preprocessing and unsupervised learning stage: Integrating multimodal AMS datasets, train-test split and applying K-Means clustering (K=4) to identify distinct physiological and AMS acclimatization profiles. 
<img width="560" height="277" alt="Stage 1" src="https://github.com/user-attachments/assets/76b4d8a0-618d-42a1-8b0f-5bac8b395e3a" />
<br>
**Stage 2 of  Machine Learning-Based Genetic Analysis of AMS pipeline** : The top five genes identified from each of the four K-Means clusters, resulting in 20 selected candidate genes associated with distinct AMS and acclimatization profiles.
<img width="601" height="208" alt="Stage 2" src="https://github.com/user-attachments/assets/7e7e0882-7247-42f1-a969-eb8704347ef5" />
<br>
**Stage 3 of  Machine Learning-Based Genetic Analysis of AMS pipeline** : The Naive Bayes classifier achieved 99.0% accuracy, with 96.15% precision, 100% recall, 98.04% F1-score, 98.67% specificity and an AUC of 1.00 on the test set.
<img width="392" height="125" alt="Stage 3" src="https://github.com/user-attachments/assets/6d5051bb-1911-45a8-91fb-e26924d76c43" />

# Quick Setup

### Create virtual environment
python -m venv venv

### Activate environment - Windows
venv\Scripts\activate

### Activate environment - macOS/Linux
source venv/bin/activate

### Install dependencies
pip install -r requirements.txt

### Run Training
python train.py

 # 📄 License

 This was the **second** project which had been independently developed by me during my two-months research internship at the **Defence Research and Development Organisation (DRDO), Ministry of Defence, Government of India**. The project is intended for research and educational purposes. **This repository does not represent an official DRDO publication, endorsement or release.**

 Copyright © 2026 **Vamika Arya**

---

 # 👩🏻‍💻 Author

 **Vamika Arya**

 <p align="center">
  <a href="https://www.linkedin.com/in/vamika-arya-4a0179288/">
    <img src="https://img.shields.io/badge/LinkedIn-Vamika%20Arya-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
</p>
