
# Run the application with Uvicorn using the command below
# uvicorn main:app --reload

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import torch
import PyPDF2
from io import BytesIO
import pickle
from fastapi import Request
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import os

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# ==================== IMPORTANT CHANGES FOR NON-PRETRAINED BERT ====================

# Load the custom tokenizer you trained from scratch
tokenizer = BertTokenizer.from_pretrained('./patient_model')

# IMPORTANT: When loading the model, you need to load the configuration first
# or ensure the model architecture matches what was trained

# Option 1: If you have a config.json in your patient_model folder
model = BertForSequenceClassification.from_pretrained('./patient_model')

# Option 2: If you need to specify custom config (use if Option 1 fails)
# config_path = "./patient_model/config.json"
# if os.path.exists(config_path):
#     model = BertForSequenceClassification.from_pretrained('./patient_model')
# else:
#     # Create config matching your training
#     config = BertConfig(
#         vocab_size=tokenizer.vocab_size,
#         hidden_size=768,
#         num_hidden_layers=12,
#         num_attention_heads=12,
#         intermediate_size=3072,
#         hidden_act="gelu",
#         hidden_dropout_prob=0.1,
#         attention_probs_dropout_prob=0.1,
#         max_position_embeddings=512,
#         type_vocab_size=2,
#         initializer_range=0.02,
#         layer_norm_eps=1e-12,
#         pad_token_id=0,
#         num_labels=20  # Adjust based on your number of classes
#     )
#     model = BertForSequenceClassification(config)
#     # Load the trained weights
#     model.load_state_dict(torch.load('./patient_model/pytorch_model.bin'))

# Load label encoder
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

# ==================== REST OF THE CODE (mostly unchanged) ====================

app = FastAPI()
templates = Jinja2Templates(directory="templates")

diseaseList = {
    "Peptic Ulcer Disease": {
        "description": "A sore that develops on the lining of the esophagus, stomach, or small intestine.",
        "medicines": ["Omeprazole", "Pantoprazole", "Ranitidine", "Esomeprazole", "Amoxicillin"],
        "specialists": ["Gastroenterologist", "General Physician", "Internal Medicine Specialist"]
    },
    "Type 2 Diabetes Mellitus": {
        "description": "A chronic condition that affects the way the body processes blood sugar (glucose).",
        "medicines": ["Metformin", "Glipizide", "Insulin", "Sitagliptin", "Canagliflozin"],
        "specialists": ["Endocrinologist", "Diabetologist", "Nutritionist"]
    },
    "Acute Myocardial Infarction": {
        "description": "A medical emergency where the blood flow to the heart is blocked.",
        "medicines": ["Aspirin", "Clopidogrel", "Statins", "Beta Blockers", "ACE Inhibitors"],
        "specialists": ["Cardiologist", "Emergency Medicine Specialist"]
    },
    "Chronic Obstructive Pulmonary Disease": {
        "description": "A group of lung diseases that block airflow and make breathing difficult.",
        "medicines": ["Tiotropium", "Albuterol", "Ipratropium", "Fluticasone", "Salmeterol"],
        "specialists": ["Pulmonologist", "General Physician", "Respiratory Therapist"]
    },
    "Cerebrovascular Accident (Stroke)": {
        "description": "A condition caused by the interruption of blood flow to the brain.",
        "medicines": ["Alteplase", "Aspirin", "Clopidogrel", "Warfarin", "Atorvastatin"],
        "specialists": ["Neurologist", "Rehabilitation Specialist", "Neurosurgeon"]
    },
    "Deep Vein Thrombosis": {
        "description": "A blood clot forms in a deep vein, usually in the legs.",
        "medicines": ["Warfarin", "Heparin", "Apixaban", "Dabigatran", "Rivaroxaban"],
        "specialists": ["Hematologist", "Vascular Surgeon", "Cardiologist"]
    },
    "Chronic Kidney Disease": {
        "description": "The gradual loss of kidney function over time.",
        "medicines": ["Erythropoietin", "Phosphate Binders", "ACE Inhibitors", "Diuretics", "Calcitriol"],
        "specialists": ["Nephrologist", "Dietitian", "Internal Medicine Specialist"]
    },
    "Community-Acquired Pneumonia": {
        "description": "A lung infection acquired outside of a hospital setting.",
        "medicines": ["Amoxicillin", "Azithromycin", "Clarithromycin", "Ceftriaxone", "Levofloxacin"],
        "specialists": ["Pulmonologist", "Infectious Disease Specialist", "General Physician"]
    },
    "Septic Shock": {
        "description": "A severe infection leading to dangerously low blood pressure.",
        "medicines": ["Norepinephrine", "Vancomycin", "Meropenem", "Hydrocortisone", "Dopamine"],
        "specialists": ["Intensivist", "Infectious Disease Specialist", "Emergency Medicine Specialist"]
    },
    "Rheumatoid Arthritis": {
        "description": "An autoimmune disorder causing inflammation in joints.",
        "medicines": ["Methotrexate", "Sulfasalazine", "Hydroxychloroquine", "Adalimumab", "Etanercept"],
        "specialists": ["Rheumatologist", "Orthopedic Specialist", "Physical Therapist"]
    },
    "Congestive Heart Failure": {
        "description": "A chronic condition where the heart doesn't pump blood effectively.",
        "medicines": ["ACE Inhibitors", "Beta Blockers", "Diuretics", "Spironolactone", "Digoxin"],
        "specialists": ["Cardiologist", "General Physician", "Cardiac Surgeon"]
    },
    "Pulmonary Embolism": {
        "description": "A blockage in one of the pulmonary arteries in the lungs.",
        "medicines": ["Heparin", "Warfarin", "Alteplase", "Rivaroxaban", "Dabigatran"],
        "specialists": ["Pulmonologist", "Hematologist", "Emergency Medicine Specialist"]
    },
    "Sepsis": {
        "description": "A life-threatening organ dysfunction caused by a dysregulated immune response to infection.",
        "medicines": ["Vancomycin", "Meropenem", "Piperacillin-Tazobactam", "Cefepime", "Dopamine"],
        "specialists": ["Infectious Disease Specialist", "Intensivist", "Emergency Medicine Specialist"]
    },
    "Liver Cirrhosis": {
        "description": "A late-stage liver disease caused by liver scarring and damage.",
        "medicines": ["Spironolactone", "Furosemide", "Lactulose", "Nadolol", "Rifaximin"],
        "specialists": ["Hepatologist", "Gastroenterologist", "Nutritionist"]
    },
    "Acute Renal Failure": {
        "description": "A sudden loss of kidney function.",
        "medicines": ["Diuretics", "Dopamine", "Calcium Gluconate", "Sodium Bicarbonate", "Epoetin"],
        "specialists": ["Nephrologist", "Critical Care Specialist", "Internal Medicine Specialist"]
    },
    "Urinary Tract Infection": {
        "description": "An infection in any part of the urinary system.",
        "medicines": ["Nitrofurantoin", "Ciprofloxacin", "Amoxicillin-Clavulanate", "Trimethoprim-Sulfamethoxazole", "Cephalexin"],
        "specialists": ["Urologist", "General Physician", "Infectious Disease Specialist"]
    },
    "Hypertension": {
        "description": "A condition in which the force of the blood against the artery walls is too high.",
        "medicines": ["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide", "Metoprolol"],
        "specialists": ["Cardiologist", "General Physician", "Nephrologist"]
    },
    "Asthma": {
        "description": "A condition in which the airways narrow and swell, causing difficulty in breathing.",
        "medicines": ["Albuterol", "Fluticasone", "Montelukast", "Budesonide", "Salmeterol"],
        "specialists": ["Pulmonologist", "Allergist", "General Physician"]
    },
    "Gastroesophageal Reflux Disease (GERD)": {
        "description": "A digestive disorder where stomach acid irritates the esophagus.",
        "medicines": ["Omeprazole", "Esomeprazole", "Ranitidine", "Lansoprazole", "Pantoprazole"],
        "specialists": ["Gastroenterologist", "General Physician", "Dietitian"]
    }
}

def textCleaning(text):
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predictTheDisease(patient_record, model, tokenizer, label_encoder):
    patient_record = textCleaning(patient_record)
    inputs = tokenizer(patient_record, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_label = torch.argmax(logits, dim=1).item()
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]
    
    # Get prediction probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence = probabilities[0][predicted_label].item()
    
    return predicted_disease, confidence

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def get_disease_details(disease_name):
    # Handle variations in disease names
    if disease_name in diseaseList:
        return diseaseList[disease_name]
    
    # Try to match with close names
    for key in diseaseList.keys():
        if disease_name.lower() in key.lower() or key.lower() in disease_name.lower():
            return diseaseList[key]
    
    return {
        "description": "No details available for this disease.",
        "medicines": [],
        "specialists": []
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = ""
        
        if file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.filename.endswith(".txt"):
            text = content.decode("utf-8")
        else:
            return JSONResponse(
                content={"error": "Unsupported file format. Please upload PDF or TXT."},
                status_code=400
            )
        
        if not text.strip():
            return JSONResponse(
                content={"error": "The uploaded file appears to be empty or couldn't be read."},
                status_code=400
            )
        
        predicted_disease, confidence = predictTheDisease(text, model, tokenizer, label_encoder)
        disease_details = get_disease_details(predicted_disease)
        
        return JSONResponse(content={
            "predicted_disease": predicted_disease,
            "confidence": f"{confidence:.2%}",
            "description": disease_details["description"],
            "medicines": disease_details["medicines"],
            "specialists": disease_details["specialists"]
        })
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"},
            status_code=500
        )

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)