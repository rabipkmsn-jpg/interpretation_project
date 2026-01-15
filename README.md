# 🏥 MEDPREDICT
### Diagnosis Prediction from Clinical Notes using BERT 🧬

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

> **A state-of-the-art AI tool that leverages Deep Learning to analyze clinical notes and predict potential medical diagnoses.**

---

## 🩺 Overview
**MedPredict** implements a powerful **BERT-based model** (`bert-base-cased`) fine-tuned for multi-class classification. By processing unstructured clinical notes and patient histories, it provides predictive insights to assist medical professionals.

---

## ✨ Key Features

* **🧹 Data Cleaning & Preprocessing**
    * *Sanitizes complex medical text for optimal analysis.*
* **🧩 Smart Tokenization**
    * *Utilizes `BertTokenizer` to convert clinical text into machine-readable formats.*
* **🧠 BERT Fine-Tuning**
    * *Adapts the pre-trained BERT architecture specifically for multi-class diagnosis.*
* **📊 Performance Evaluation**
    * *Detailed metrics including Confusion Matrices and Classification Reports.*
* **💾 Model Deployment**
    * *Saves trained models for seamless integration with FastAPI.*

---

## 📉 Evaluation Metrics
We ensure reliability through rigorous testing:
* **Confusion Matrix** 🏳️ — *Visualizes true vs. predicted classifications.*
* **Classification Report** 📋 — *Assess Precision, Recall, and F1-Scores.*

---

## 🚀 How to Run

Follow these steps to get MedPredict up and running on your local machine.
1. 📥 **Clone the repository:**  
   ```bash
   git clone https://github.com/aykaimran/MedPredict.git
   cd MedPredict
2. 🏋️‍♀️ Train and Save the Model:
   - Open the MedPredict.ipynb notebook.
   - Run the cells to train the model and save it locally.
3. 📂 Add the Pre-trained Model:
   - Download the saved model files.
   - Place them inside the MedPredict folder.
4. 📦 Install Dependencies:
   - Install the required Python packages listed at the top of main.py.
    ```bash
    pip install fastapi uvicorn torch transformers PyPDF2 aiofiles scikit-learn nltk python-multipart
5. ⚡ Run the Prediction Script:
   - Execute main.py using the instructions provided at the top of the file.
   ```bash
   uvicorn main:app --reload

---

## 👩‍⚕️ Contributors
The team behind MedPredict:
- [Vaniya Ijaz](https://github.com/VE-Vaniya)
- [Ayka Imran](https://github.com/aykaimran)
- [Maryam Irshad](https://github.com/maryamirshad04)
- [Walija Fatima](https://github.com/Jee-core)

<p align="center"> <sub>Built with ❤️ and ☕ by the MedPredict Team</sub> </p>
