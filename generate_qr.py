import qrcode

# --- CONFIGURATION ---
# Replace this with your AWS Public IP address
# Make sure to include http:// and :8000
WEBSITE_URL = "https://subeffectively-premegalithic-karlee.ngrok-free.dev" 

# --- GENERATE QR CODE ---
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

qr.add_data(WEBSITE_URL)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")

# --- SAVE IMAGE ---
filename = "ar_translator_qr.png"
img.save(filename)

print(f"Success! QR code saved as '{filename}'.")
print(f"Scan it to go to: {WEBSITE_URL}")