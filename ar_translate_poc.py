import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from transformers import pipeline
import torch
import os
import warnings # ### OPTIMIZATION ### Import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning) # ### OPTIMIZATION ###

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

# 4. ### OPTIMIZATION ### Process one frame every N frames
PROCESS_EVERY_N_FRAMES = 15

# 5. ### OPTIMIZATION ### Resize frame for AI processing
PROCESSING_WIDTH = 800

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

    # ### OPTIMIZATION ### Check for GPU and set device
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("GPU is available! Running on GPU.")
        device_num = 0
    else:
        print("Using CPU. Note: This module is much faster with a GPU.")
        device_num = -1

    # 1. Load EasyOCR Reader
    #    This will download the model for the specified language.
    print(f"Loading EasyOCR for language: {OCR_LANG}")
    reader = easyocr.Reader(OCR_LANG, gpu=use_gpu) # ### OPTIMIZATION ### Pass use_gpu flag

    # 2. Load Translation Pipeline
    #    This will download the Hugging Face model.
    print(f"Loading translation model: {MODEL_NAME}")
    try:
        # ### OPTIMIZATION ### Pass device_num to pipeline
        translator = pipeline("translation", model=MODEL_NAME, device=device_num)
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

    # ### OPTIMIZATION ### Variables for frame skipping
    frame_count = 0
    last_results = []

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ### OPTIMIZATION ### Main AI processing block
        # Only run the heavy AI models every N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            
            # Clear old results
            last_results = []
            
            # --- Resize optimization ---
            orig_h, orig_w = frame.shape[:2]
            scale_factor = orig_w / PROCESSING_WIDTH
            processing_height = int(orig_h / scale_factor)
            
            # Resize the frame for AI
            small_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height))
            
            # Convert the SMALL OpenCV frame (BGR) to RGB
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # --- OCR DETECTION (on the small frame) ---
            try:
                results = reader.readtext(small_frame_rgb, batch_size=5, x_ths = 1.0)
            except Exception as e:
                print(f"Error during OCR: {e}")
                continue

            for (bbox, text, prob) in results:
                if prob < 0.2: # Confidence threshold
                    continue

                # --- TRANSLATION ---
                try:
                    translated_obj = translator(text, max_length=128)
                    translated_text = translated_obj[0]['translation_text']
                except Exception as e:
                    print(f"Error during translation: {e}")
                    translated_text = "[Translation Error]"

                # --- STORE RESULTS ---
                
                # ### OPTIMIZATION ### Scale bounding box back up to original size
                (tl, tr, br, bl) = bbox
                
                tl_orig = (int(tl[0] * scale_factor), int(tl[1] * scale_factor))
                br_orig = (int(br[0] * scale_factor), int(br[1] * scale_factor))
                
                # Store the data we need for drawing
                last_results.append((tl_orig, br_orig, translated_text))


        # --- DRAWING (Runs on EVERY frame) ---
        # This part is fast, so we do it always, using the last known results.
        
        # Convert FULL, ORIGINAL frame (numpy array) to PIL Image for drawing
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Draw all the *last known* results
        for (tl, br, translated_text) in last_results:
            
            # Draw a white rectangle to cover the original text
            draw.rectangle([tl, br], fill="white", outline="white")

            # Draw the translated text on top
            text_position = (tl[0], tl[1] - FONT_SIZE)
            if text_position[1] < 0:
                text_position = (tl[0], 0)
                
            draw.text(text_position, font=font, text=translated_text, fill="black")

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