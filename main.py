import os
import html
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google.cloud import vision
from google.cloud import translate_v2 as translate
from ddgs import DDGS
from google.oauth2 import service_account


# --- AUTHENTICATION ---

creds_json_str = os.environ['GOOGLE_APPLICATION_CREDENTIALS'] 

if creds_json_str:
    # Create credentials object from the secret JSON string
    creds_dict = json.loads(creds_json_str)
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    print("Credentials loaded from Environment Secret.")
else:
    # Fallback for local testing
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS secret not found. Trying local file.")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
    credentials = None # Client libraries will look for env var automatically

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


# --- HELPER: Web Search ---
def get_english_definition(term):
    """
    Searches for the dictionary definition of an English word.
    e.g. term="Year" -> returns "Time taken by the earth to orbit the sun..."
    """
    queries = [
        f"define {term} dictionary",
        f"meaning of {term} in english",
        f"what is a {term}"
    ]
    try:
        with DDGS() as ddgs:
            for q in queries:
                results = list(ddgs.text(q, max_results=1))
                if results:
                    return results[0]['body']
    except: pass
    return None

@app.post("/translate/")
async def translate_image(
    file: UploadFile = File(...),
    target_lang: str = Form("en")
):
    try:
        contents = await file.read()
        image = vision.Image(content=contents)
        response = vision_client.document_text_detection(image=image)
        
        processed_results = []
        word_data = []

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    paragraph_text = ""
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        paragraph_text += word_text + " "
                        w_verts = word.bounding_box.vertices
                        word_data.append({
                            "text": word_text,
                            "tl": [w_verts[0].x, w_verts[0].y],
                            "br": [w_verts[2].x, w_verts[2].y]
                        })

                    paragraph_text = paragraph_text.strip()
                    if len(paragraph_text) < 2: continue

                    translated_text = "[Error]"
                    try:
                        translation = translate_client.translate(
                            paragraph_text, 
                            target_language=target_lang, 
                            model="nmt"
                        )
                        translated_text = html.unescape(translation['translatedText'])
                    except: pass

                    vertices = paragraph.bounding_box.vertices
                    processed_results.append({
                        "tl": [vertices[0].x, vertices[0].y],
                        "br": [vertices[2].x, vertices[2].y],
                        "text": translated_text
                    })

        return { "translations": processed_results, "words": word_data }
    except Exception as e:
        return {"error": str(e)}

@app.get("/define/")
async def define_word(word: str, target_lang: str = "en", source_lang: str = "auto"):
    # Just remove whitespace.
    clean_word = word.strip()
    
    # 1. Language Setup
    lang_map = {'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French'}
    source_name = lang_map.get(source_lang, 'Foreign Language')
    target_name = lang_map.get(target_lang, 'English')

    print(f"Processing: {clean_word} ({source_name} -> {target_name})")

    # 2. STEP 1: Translate
    translated_word = clean_word
    english_meaning = ""
    
    try:
        # Get English meaning first (The "Bridge")
        trans = translate_client.translate(
            clean_word, 
            target_language='en',
            source_language=source_lang if source_lang != 'auto' else None
        )
        english_meaning = html.unescape(trans['translatedText'])
        
        # If user wants target lang (e.g. Spanish), translate "Year" -> "Año"
        if target_lang != 'en':
            trans_target = translate_client.translate(english_meaning, target_language=target_lang)
            translated_word = html.unescape(trans_target['translatedText'])
        else:
            translated_word = english_meaning

    except Exception as e:
        print(f"Translation Error: {e}")
        return {"word": clean_word, "definition": "Could not translate."}

    # 3. STEP 2: Search Definition
    definition_text = get_english_definition(english_meaning)
    
    if not definition_text:
        # Fallback: If English search fails, try the original word
        definition_text = get_english_definition(clean_word)

    # Translate definition if needed
    if definition_text and target_lang != 'en':
        try:
            t_def = translate_client.translate(definition_text, target_language=target_lang)
            definition_text = html.unescape(t_def['translatedText'])
        except: pass

    # 4. STEP 3: Format Output
    # Format: "The Hindi word 'साल' translates to 'Year'..."
    final_output = f"{translated_word}\n\n{definition_text or 'Definition not found.'}"
    
    if english_meaning.lower() != clean_word.lower():
         final_output = (
            f"The {source_name} word '{clean_word}' translates to "
            f"'{translated_word}' in {target_name}.\n\n"
            f"{translated_word}: {definition_text or 'Definition not found.'}"
        )

    return {
        "word": translated_word, 
        "definition": final_output
    }

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html missing</h1>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)