import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from transformers import pipeline
import torch
import os

# --- CONFIGURATION ---

# 1. Translation Model (Source -> Target)
#    You can find model names here: https://huggingface.co/models?pipeline_tag=translation
#    Example: 'Helsinki-NLP/opus-mt-en-fr' (English to French)
#    Example: 'Helsinki-NLP/opus-mt-es-en' (Spanish to English)
SOURCE_LANG = 'en'
TARGET_LANG = 'es' # Spanish
MODEL_NAME = f'Helsinki-NLP/opus-mt-{SOURCE_LANG}-{TARGET_LANG}'

# 2. Font for displaying translated text
#    OpenCV's built-in fonts don't support many languages.
#    We will use Pillow to draw text. You MUST provide a .ttf font file.
#    Download one (e.g., "NotoSans-Regular.ttf") and place it in the same folder.
#    You can get Noto fonts here: https://fonts.google.com/noto
FONT_FILE = "NotoSans-VariableFont_wdth,wght.ttf"
DEFAULT_FONT = "arial.ttf" # Fallback if your font isn't found
FONT_SIZE = 20

# 3. EasyOCR Language
#    This is the language it will *detect*.
OCR_LANG = [SOURCE_LANG] 

# --- END CONFIGURATION ---

def get_font(font_file, size):
    """Loads the specified font, with a fallback to a system default."""
    if os.path.exists(font_file):
        try:
            return ImageFont.truetype(font_file, size)
        except IOError:
            print(f"Error loading font '{font_file}'. Trying default.")
    
    # Fallback
    try:
        return ImageFont.truetype(DEFAULT_FONT, size)
    except IOError:
        print(f"Error loading default font '{DEFAULT_FONT}'. Using tiny built-in font.")
        return ImageFont.load_default()

def main():
    print("Loading models... This may take a few minutes on the first run.")

    # 1. Load EasyOCR Reader
    #    This will download the model for the specified language.
    print(f"Loading EasyOCR for language: {OCR_LANG}")
    reader = easyocr.Reader(OCR_LANG, gpu=torch.cuda.is_available())

    # 2. Load Translation Pipeline
    #    This will download the Hugging Face model.
    print(f"Loading translation model: {MODEL_NAME}")
    try:
        translator = pipeline("translation", model=MODEL_NAME)
    except Exception as e:
        print(f"Error loading translation model: {e}")
        print("Please ensure the model name is correct and you have an internet connection for the first run.")
        return

    # 3. Load Font
    print(f"Loading font: {FONT_FILE}")
    font = get_font(FONT_FILE, FONT_SIZE)

    # 4. Start Video Capture
    print("Starting webcam feed...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Setup complete. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the OpenCV frame (BGR) to a format EasyOCR likes (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- OCR DETECTION ---
        # 'detail=0' gives faster, simpler output (just text)
        # 'detail=1' gives bounding boxes, text, and confidence
        try:
            results = reader.readtext(frame_rgb)
        except Exception as e:
            print(f"Error during OCR: {e}")
            continue

        # Convert OpenCV frame (numpy array) to PIL Image for drawing
        # This allows us to draw non-ASCII characters
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        for (bbox, text, prob) in results:
            if prob < 0.4: # Confidence threshold
                continue

            # --- TRANSLATION ---
            try:
                translated_obj = translator(text, max_length=50)
                translated_text = translated_obj[0]['translation_text']
            except Exception as e:
                print(f"Error during translation: {e}")
                translated_text = "[Translation Error]"

            # --- DRAWING ---
            # Get the top-left and bottom-right coordinates from the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))

            # Draw a white rectangle to cover the original text
            draw.rectangle([tl, br], fill="white", outline="white")

            # Draw the translated text on top
            # We move it up slightly to sit nicely
            text_position = (tl[0], tl[1] - FONT_SIZE)
            draw.text(text_position, translated_text, font=font, fill="black")
            
            # Optional: Draw the original bounding box
            # draw.rectangle([tl, br], outline="red", width=2)
            # draw.text(tl, text, font=font, fill="red")


        # Convert the PIL Image back to an OpenCV frame (numpy array)
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow('Real-time AR Translator (Proof-of-Concept)', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Stream stopped.")

if __name__ == "__main__":
    main()