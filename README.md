# DSFv0 Raspberry Pi 5 Deployment Guide

This guide explains how to **deploy the DSFv0 sentiment analysis model** (saved as `best.onnx`) on a **Raspberry Pi 5**, and run it **live using the Raspberry Pi camera**. It extracts text from the camera view using OCR and displays real-time sentiment predictions on screen.

> ❗ This version uses **text captured from the camera only** (no voice input or audio-based processing).

---

## 🧰 Requirements

- Raspberry Pi 5 with Raspberry Pi OS (64-bit recommended)
- Raspberry Pi Camera Module (connected and enabled)
- MicroSD Card (32GB+)
- Internet connection
- `best.onnx` model (already trained)

---

## 📦 Step 1: Update Raspberry Pi and Enable Camera

Open a terminal and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo raspi-config
```

Then:

1. Go to **Interface Options**
2. Select **Camera**
3. Choose **Enable**
4. Reboot the Pi

---

## 🧱 Step 2: Install Dependencies

Install system packages:

```bash
sudo apt install python3-pip python3-opencv libatlas-base-dev tesseract-ocr -y
```

Install Python libraries:

```bash
pip3 install --upgrade pip
pip3 install numpy opencv-python pillow onnxruntime pytesseract transformers torch
```

---

## 📂 Step 3: Set Up Project Files

Clone the DSFv0 repo and move into it:

```bash
cd ~
git clone https://github.com/ShehabElsheikh/DSFv0.git
cd DSFv0
```

Now, place your `best.onnx` file into the DSFv0 folder:

```bash
# Move your ONNX model here
mv /path/to/best.onnx ./best.onnx
```

---

## 📸 Step 4: Create the Live Camera Script

Create a new Python file named `camera_onnx.py` inside the `DSFv0` folder:

```bash
nano camera_onnx.py
```

Paste this code inside and save (`CTRL+O`, `ENTER`, then `CTRL+X` to exit):

```python
import cv2
import pytesseract
import onnxruntime
import numpy as np
from transformers import BertTokenizer

# Load tokenizer and ONNX model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
session = onnxruntime.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

def preprocess(text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors='np')
    return {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }

def infer(text):
    inputs = preprocess(text)
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0])
    return "Positive" if pred == 1 else "Negative"

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error.")
    exit()

print("[INFO] Starting live camera. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to text
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    # Predict sentiment if text is detected
    if text.strip():
        sentiment = infer(text)
        cv2.putText(frame, f"{sentiment}: {text.strip()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show frame
    cv2.imshow("Sentiment Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ▶️ Step 5: Run the Script

To start live sentiment detection using your camera:

```bash
python3 camera_onnx.py
```

- The camera will open and scan for visible text.
- If it detects readable text, it will display:
  - **"Positive"** or **"Negative"** sentiment
  - Along with the captured text
- Press `Q` to quit the program.

---

## ✅ Final Notes

- Works best when pointed at **printed or digital subtitles, labels, or readable sentences**.
- If the camera doesn't open, make sure it's enabled and connected properly.
- Make sure your `best.onnx` matches the tokenizer (`bert-base-uncased`).

---

## 👨‍💻 Author

Built by [Shehab ElSheikh](https://github.com/ShehabElsheikh)  
Email: shehabdiaa12345@gmail.com

---
