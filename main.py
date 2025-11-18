import cv2
import numpy as np
import easyocr
from transformers import pipeline
import torch
import os
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # Import for serving HTML
from PIL import Image

# --- CONFIGURATION ---
SOURCE_LANG = 'en'
TARGET_LANG = 'es'
# MODEL_NAME = f'Helsinki-NLP/opus-mt-{SOURCE_LANG}-{TARGET_LANG}'
# This model auto-detects many languages and translates to English
MODEL_NAME = 'Helsinki-NLP/opus-mt-mul-en'
OCR_LANG = [SOURCE_LANG] 
PROCESSING_WIDTH = 800

# --- LOAD MODELS ON STARTUP ---
print("Loading models... This will happen once on startup.")
use_gpu = torch.cuda.is_available()
device_num = 0 if use_gpu else -1
if use_gpu:
    print("GPU found! Loading models to GPU.")
else:
    print("No GPU found. Loading models to CPU.")

reader = easyocr.Reader(OCR_LANG, gpu=use_gpu)
print("EasyOCR model loaded.")

translator = pipeline(
    "translation", 
    model=MODEL_NAME, 
    device=device_num,
    num_beams=4
)
print("Translation model loaded.")
print("Backend is ready and waiting for requests.")

# --- INITIALIZE FASTAPI APP ---
app = FastAPI(title="AR Translation Backend")

# Add CORS middleware to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- API ENDPOINT for translation ---
@app.post("/translate/")
async def translate_image(file: UploadFile = File(...)):
    """Receives an image, performs OCR/Translation, and returns JSON."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run OCR
        results = reader.readtext(
            frame_rgb, 
            batch_size=5, 
            x_ths=1.5,
            # y_ths=0.5
        )
        
        processed_results = []
        for (bbox, text, prob) in results:
            if prob < 0.4:
                continue

            # Run Translation
            try:
                translated_obj = translator(text, max_length=128)
                translated_text = translated_obj[0]['translation_text']
            except Exception as e:
                print(f"Translation error: {e}")
                translated_text = "[Error]"

            (tl, tr, br, bl) = bbox
            tl_simple = [int(tl[0]), int(tl[1])]
            br_simple = [int(br[0]), int(br[1])]
            
            processed_results.append({
                "tl": tl_simple,
                "br": br_simple,
                "text": translated_text
            })

        return processed_results
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e)}

# --- API ENDPOINT for serving the webpage ---~
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serves the frontend HTML file."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Make sure index.html is in the same folder as main.py.</p>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    # This runs the whole app
    uvicorn.run(app, host="127.0.0.1", port=8000)