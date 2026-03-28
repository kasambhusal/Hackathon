from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, pipeline
import librosa
import torch
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Dementia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- FIX START ---
model_id = "Mrsmetamorphosis/dementia-wav2vec-scientific-specaugment-V2"

print("Loading AI Model and Manual Feature Extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

model = AutoModelForAudioClassification.from_pretrained(model_id)

classifier = pipeline(
    "audio-classification", 
    model=model, 
    feature_extractor=feature_extractor
)

print("Model loaded successfully!")

@app.post("/predict")
async def predict_dementia(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        audio_data = await file.read()
        buffer.write(audio_data)
        
    try:
        audio_array, sampling_rate = librosa.load(temp_file_path, sr=16000)
        
        # Running prediction
        predictions = classifier(audio_array)
        
        top_prediction = predictions[0]
        label = top_prediction["label"].lower()
        
        has_dementia = "dementia" in label and "no" not in label
        confidence_percent = f"{top_prediction['score'] * 100:.2f}%"

        return {
            "filename": file.filename,
            "has_dementia": has_dementia,
            "confidence_percentage": confidence_percent,
            "detailed_analysis": predictions
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)