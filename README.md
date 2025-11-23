# 📷 AR Translator

AR Translator is a real-time Augmented Reality web application that translates text from your camera feed instantly. It leverages Google Cloud Vision for OCR and Google Cloud Translation API for context-aware translation.

Additionally, it features a "Smart Dictionary Mode" that provides definitions for tapped words, handling transliterations (e.g., "bhai" -> "Brother") and multi-language support seamlessly.

# 🚀 Features

Real-time AR Translation: Point your camera at any text to see it translated and overlaid instantly.

Smart Dictionary: Tap any word on the screen to fetch its definition.

Supports English definitions via Merriam-Webster API.

Handles transliterated words 

Provides definitions in the target language (e.g., Hindi definition for English words).

Multi-Language Support: Supports English, Hindi, Spanish, French, and more via Google Cloud.

Mobile Optimized: Full-screen, responsive camera view with no scrolling or black bars.

Secure Backend: Uses a FastAPI proxy to securely manage API keys.

# 🛠️ Tech Stack

Frontend: HTML5, CSS3, JavaScript (Native Camera API).

Backend: Python, FastAPI (for API routing).

AI/ML Services:

Google Cloud Vision API: For Optical Character Recognition (OCR).

Google Cloud Translation API (V2/V3): For Neural Machine Translation (NMT).

Dictionary: Merriam-Webster Collegiate Dictionary API.

Deployment: Docker, Hugging Face Spaces, or any cloud provider (AWS/Render).

# 📦 Prerequisites

Before running this project, ensure you have the following:

Python 3.10+ installed.

Google Cloud Account with a project enabling:

Cloud Vision API

Cloud Translation API

Merriam-Webster API Key (Free tier is sufficient).

Git installed.

# 🔧 Installation & Setup

1. Clone the Repository

git clone https://github.com/Rohithiwal/ar-translator.git
cd ar-translator


2. Install Dependencies

Create a virtual environment (optional but recommended) and install required packages:

pip install -r requirements.txt


3. Setup Credentials

To run the backend, you need to configure your API keys:

Google Cloud:

Download your Service Account Key JSON file from Google Cloud Console.

Rename it to credentials.json and place it in the root directory of the project.

Merriam-Webster:

Open main.py.

# ▶️ Running the Application

Local Development

Start the backend server using Uvicorn:

uvicorn main:app --host 0.0.0.0 --port 8000

The server will start at http://0.0.0.0:8000.

# 📂 Project Structure

main.py: The FastAPI backend handling OCR, translation, and dictionary lookups.

index.html: The frontend UI with camera logic and AR overlay drawing.

requirements.txt: List of Python dependencies.

Dockerfile: Configuration for building the Docker image.

credentials.json: (Ignored by Git) Your Google Cloud service account key.

# 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# 📄 License

This project is licensed under the MIT License.
