import os
import html
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google.cloud import vision
from google.cloud import translate_v2 as translate

# --- CONFIGURATION ---
TARGET_LANG = 'en'

# --- AUTHENTICATION ---
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

# Initialize Clients
print("Initializing Google Cloud Clients...")
try:
    vision_client = vision.ImageAnnotatorClient()
    translate_client = translate.Client()
    print("SUCCESS: Google Clients are ready.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/translate/")
async def translate_image(file: UploadFile = File(...)):
    try:
        # 1. Read Image
        contents = await file.read()
        image = vision.Image(content=contents)
        
        # 2. Perform Document Text Detection (Better for sentences)
        response = vision_client.document_text_detection(image=image)
        
        processed_results = []
        
        # Navigate the hierarchy: Page -> Block -> Paragraph
        # We will translate at the BLOCK or PARAGRAPH level to keep context.
        
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    
                    # 3. Reconstruct the Sentence from words
                    paragraph_text = ""
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        paragraph_text += word_text + " "
                    
                    paragraph_text = paragraph_text.strip()
                    
                    # Skip empty or tiny noise
                    if len(paragraph_text) < 2: continue

                    # 4. Translate the WHOLE paragraph (Context preserved!)
                    try:
                        translation = translate_client.translate(
                            paragraph_text, 
                            target_language=TARGET_LANG,
                            model="nmt"
                        )
                        translated_text = html.unescape(translation['translatedText'])
                    except:
                        translated_text = "[Error]"

                    # 5. Get Coordinates for the whole Paragraph
                    # The bounding box is stored in paragraph.bounding_box
                    vertices = paragraph.bounding_box.vertices
                    
                    processed_results.append({
                        "tl": [vertices[0].x, vertices[0].y], # Top-Left
                        "br": [vertices[2].x, vertices[2].y], # Bottom-Right
                        "text": translated_text
                    })

        return processed_results
        
    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html missing</h1>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)