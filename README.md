# MEDPREDICT - Diagnosis Prediction from Clinical Notes using BERT 

This project implements a BERT-based model to predict potential medical diagnoses from clinical notes and patient histories. It leverages the `bert-base-cased` pre-trained model and fine-tunes it for multi-class classification.  

---

## Features

- **Data Cleaning & Preprocessing:** Cleans text data and prepares it for modeling.  
- **Tokenization:** Converts text into tokens compatible with BERT using `BertTokenizer`.  
- **Fine-tuning BERT:** Adapts the pre-trained BERT model for multi-class diagnosis prediction.  
- **Model Evaluation:** Assesses performance using a confusion matrix and classification report.  
- **Model Saving & Deployment:** Saves the trained model and tokenizer for future predictions.  

---

## Evaluation
The model performance can be evaluated using:
- **Confusion Matrix** – to visualize classification results.
- **Classification Report** – to assess precision, recall, and F1-score.

---

## To run
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/aykaimran/MedPredict.git
   cd MedPredict
2. Train and Save the Model:
   - Open the MedPredict.ipynb notebook.
   - Run the cells to train the model and save it locally.
3. Add the Pre-trained Model:
   - Download the saved model files.
   - Place them inside the MedPredict folder.
4. Install Dependencies:
   - Install the required Python packages listed at the top of main.py.
    ```bash
    pip install fastapi uvicorn torch transformers PyPDF2 aiofiles scikit-learn nltk python-multipart
5. Run the Prediction Script:
   - Execute main.py using the instructions provided at the top of the file.
   ```bash
   uvicorn main:app --reload

---

## Contributors
- [Vaniya Ijaz](https://github.com/VE-Vaniya)
- [Ayka Imran](https://github.com/aykaimran)
- [Maryam Irshad](https://github.com/maryamirshad04)
- [Walija Fatima](https://github.com/Jee-core)
