# 🩺 Clinical Diagnosis Prediction from Patient Records using BERT

## Introduction
With the rapid growth of electronic health records (EHRs), a large amount of unstructured clinical text is generated daily. Extracting meaningful insights from these records can assist healthcare professionals in decision-making. This project focuses on leveraging Natural Language Processing (NLP) and deep learning techniques to predict possible medical diagnoses from patient clinical notes.

---

## 📝 Project Description
This project implements a high-performance **Medical Diagnosis Prediction System** powered by a **custom BERT architecture**. Unlike typical solutions, this model was **initialized and trained from scratch** specifically to master the complex vocabulary of clinical medical records.

**Key Features:**
* **Custom-Trained BERT:** Built from the ground up to capture nuanced medical context.
* **FastAPI Backend:** Provides a robust and fast API for real-time inference.
* **Multi-Format Support:** Seamlessly processes patient records in both **PDF** and **TXT** formats.
* **Comprehensive Output:** Delivers the most likely diagnosis along with:
    * Confidence Scores 📊
    * Detailed Disease Descriptions 📖
    * Recommended Medications 💊
    * Specialist Referrals 👨‍⚕️
    
The complete pipeline includes:
- Data preprocessing and cleaning
- Custom tokenizer training
- Model training and evaluation
- Model saving and deployment for inference

---

## Objective
The main objectives of this project are:
- To preprocess and normalize unstructured clinical notes
- To convert clinical text into model-readable token sequences
- To train a BERT-based model to predict patient diagnoses
- To evaluate the model using standard classification metrics
- To build a reusable prediction pipeline for real-world inference

---

## Workflow
1. **Prepare and Clean the Data**
   - Lowercasing text
   - Removing numbers, punctuation, and stopwords
2. **Tokenization**
   - Train a custom WordPiece tokenizer on clinical notes
3. **Model Training**
   - Train a BERT model from scratch for diagnosis prediction
4. **Model Evaluation**
   - Confusion Matrix
   - Classification Report
   - Accuracy and F1 Score
5. **Deployment**
   - Save trained model, tokenizer, and label encoder
   - Build a prediction function for new patient records

---

## Tech Stack
- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - PyTorch
  - Hugging Face Transformers
  - Tokenizers
  - Scikit-learn
  - Pandas & NumPy
  - NLTK
  - Matplotlib & Seaborn
- **Model Architecture:** BERT (Transformer-based)
- **Hardware Support:** GPU (CUDA) with mixed precision (FP16)

---

## Blockers / Hurdles Faced
- **Training BERT from Scratch:**  
  Training without pretrained weights required careful tuning of learning rate, warmup steps, and epochs.
- **Data Leakage Risk:**  
  Ensuring that diagnosis labels did not trivially appear in clinical notes required strict preprocessing.
- **Class Imbalance:**  
  Some diagnoses appeared less frequently, affecting prediction performance.
- **High Computational Cost:**  
  Long sequences and large model size required gradient checkpointing and mixed precision training.

---

## Achievements
- Successfully built a complete end-to-end NLP pipeline for clinical diagnosis prediction
- Trained a BERT-based classifier capable of handling long clinical notes
- Achieved meaningful performance evaluated through accuracy, F1 score, and confusion matrix
- Implemented a reusable inference pipeline for real-time predictions
- Saved and packaged the trained model for deployment

---

## Future Scope
- Fine-tuning pretrained clinical language models (e.g., BioBERT, ClinicalBERT)
- Incorporating attention-based explanation techniques for interpretability
- Expanding the dataset to include more diagnoses
- Deploying the model as a web or API-based service
- Integrating confidence scores for safer medical decision support

---

## Sample Prediction
```text
Input:
yearold male presents heartburn regurgitation sour taste mouth especially meals patient selfmedicating overthecounter antacids symptoms persist hour ph monitoring test confirms diagnosis gerd patient started ppi advised avoid trigger foods

Output:
Predicted Disease: Gastroesophageal Reflux Disease

## 🚀 How to Run the Project

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
 Run in terminal:
py -m uvicorn main:app --reload
