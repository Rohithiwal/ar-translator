# AR Translator

A real-time Augmented Reality translation web app that uses your camera to detect text, translate it instantly, and overlay the translation. It also features a "Dictionary Mode" where tapping a word provides its definition.

🚀 Features

Real-time AR Translation: Point your camera at text to see it translated instantly.

Smart Dictionary: Tap any word on the screen to see its definition.

Supports English definitions via Merriam-Webster API.

Handles transliterated words (e.g., "bhai" -> "Brother").

Provides definitions in the target language (e.g., Hindi definition for English words).

Multi-Language Support: Supports English, Hindi, Spanish, French, and more via Google Cloud.

Mobile Optimized: Full-screen camera view with no scrolling or black bars.

Secure: Uses a backend proxy to hide API keys.

🛠️ Tech Stack

Frontend: HTML5, CSS3, JavaScript (Native Camera API).

Backend: Python, FastAPI (for API routing).

AI/ML: * Google Cloud Vision API: For Optical Character Recognition (OCR).

Google Cloud Translation API (V2/V3): For Neural Machine Translation (NMT).

Dictionary: Merriam-Webster Collegiate Dictionary API.

📦 Prerequisites

Python 3.8+ installed.

Google Cloud Account with a project enabling:

Cloud Vision API

Cloud Translation API

Merriam-Webster API Key (Free tier is fine).

🔧 Setup & Installation

1. Clone the Repository

git clone [https://github.com/Rohithiwal/ar-translator.git](https://github.com/Rohithiwal/ar-translator.git)
cd ar-translator


2. Install Dependencies

pip install -r requirements.txt


3. Setup Credentials

Google Cloud: Place your credentials.json file in the root directory of the project.

Merriam-Webster: Open main.py and paste your API key into the MW_API_KEY variable.

4. Run the Server

# Run with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000


5. Access on Mobile

To use the camera on your phone, you need a secure connection (HTTPS). Use Ngrok to tunnel your localhost.

Install Ngrok.

Run the tunnel:

ngrok http 8000


Copy the https://....ngrok-free.dev link and open it on your mobile browser.

📂 Project Structure

main.py: The FastAPI backend handling OCR, translation, and dictionary lookups.

index.html: The frontend UI with camera logic and AR overlay drawing.

requirements.txt: List of Python dependencies.

credentials.json: (Ignored by Git) Your Google Cloud service account key.

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📄 License

MIT
