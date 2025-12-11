📌 Overview

This repository contains an engineer-grade OCR (Optical Character Recognition) and AI text-summarization pipeline designed to extract meaningful text from images that may contain hidden, stylized, or difficult-to-read fonts. The system uses a combination of:

EasyOCR (deep-learning based text extraction)

Tesseract OCR (classical OCR fallback)

OpenCV image preprocessing

PyTorch GPU acceleration (if available)

HuggingFace Transformers for AI summarization

python-docx for automated Word document generation

JSON reporting for diagnostics/debugging

The script processes all images in a specified folder, applies multi-step OCR, summarizes detected text into meaningful “advertising-style” lines, and outputs:

A professional Word document that includes:

Each image

All extracted raw text

AI-generated summarized/ad-copy text

JSON reports for every image, containing structured OCR data useful for debugging or downstream automation.

This project was built to be robust, modular, fault-tolerant, and production-ready.

⚙️ Features
✔ Multi-engine OCR Pipeline

Primary engine: EasyOCR (deep neural network text detection + recognition)

Secondary engine: Tesseract OCR (when installed, used as a fallback)

Automated confidence filtering

Multi-variant preprocessing using OpenCV

✔ AI Summarization

Uses HuggingFace Transformers with the facebook/bart-large-cnn model (or any compatible summarization model) to generate short, meaningful summaries suitable for advertising or content extraction.

✔ Image Report Generation

Each image produces a JSON file containing raw OCR results, detected text lines, and summaries.

✔ Word Document Export

The final .docx file includes:

The image

The text extracted

The summarized “meaningful text” section

This makes it easy to share, print, or manually review results.

✔ GPU Support (Optional)

If PyTorch detects CUDA, EasyOCR will run much faster.
Fallback to CPU is automatic.

📂 Project Structure
ocr_ad_pipeline.py         # Main OCR & AI processing script
/outputs                   # (User-specified) Folder where .docx and JSON files are written
/images                    # Folder containing your input images
README.md                  # Documentation (this file)

📜 Code Architecture

The script is broken into clean, maintainable sections:

1. Configuration (OCRConfig)

Defines:

languages

resize factors

confidence thresholds

summarizer model

fallback rules

This makes the pipeline highly configurable.

2. Image Preprocessing

Using OpenCV:

grayscale

CLAHE enhancement

thresholding

inverse binary

sharpening

image scaling

Each variant increases the chance of detecting difficult text.

3. OCR Extraction

Each OCR engine is wrapped in separate functions:

ocr_easyocr()

ocr_tesseract()

Both engines return extracted text lines, which are then deduplicated and cleaned.

4. AI Summarization

All OCR text is passed into a HuggingFace summarizer model to generate meaningful, condensed text.

5. Output Report Generation

The script produces:

JSON logs per image

A final Word document with images + extracted text + summaries

Uses python-docx to construct documents programmatically.

6. Error Handling

Common issues (missing GPU, missing Tesseract, unreadable images, etc.) are trapped and logged, ensuring the pipeline never crashes and always returns useful output.

📦 Libraries Used
Library	Purpose
EasyOCR	Neural-network OCR engine
Tesseract OCR	Classical OCR fallback
OpenCV (cv2)	Image preprocessing
Pillow	Image handling
transformers	AI text summarization
torch + torchvision	Neural network acceleration
python-docx	Word document creation
numpy	Image array manipulation
scikit-image	Additional image utilities (used by EasyOCR)
🧠 OCR & AI References

EasyOCR
https://github.com/JaidedAI/EasyOCR

Tesseract OCR (UB Mannheim Windows Build)
https://github.com/UB-Mannheim/tesseract/wiki

PyTorch
https://pytorch.org/

Transformers (HuggingFace)
https://huggingface.co/docs/transformers/index

facebook/bart-large-cnn model
https://huggingface.co/facebook/bart-large-cnn

OpenCV Image Processing Techniques
https://docs.opencv.org/

📌 IMPORTANT — Required Directory Changes in the Code

Inside the script, you must update the folder paths for:

1. Input image folder
image_folder = Path(r"C:\Users\YOURNAME\Desktop\...\images")

2. Output folder selection

The script prompts the user interactively:

Please enter the output directory:


The Word document + JSON logs will be created there.

3. Optional customizations:

Changing OCR languages

Changing summarizer model

Adjusting confidence thresholds

Enabling/disabling Tesseract fallback

All can be changed in the OCRConfig section.

▶️ Running the Script

Once dependencies are installed:

python ocr_ad_pipeline.py


The script will:

Ask for the output folder

Process all images

Create the .docx file

Produce JSON reports

📥 Installation Instructions (Windows)

Install required packages:

python -m pip install easyocr pytesseract opencv-python pillow python-docx transformers torch torchvision


Install Tesseract:

👉 https://github.com/UB-Mannheim/tesseract/wiki

(Be sure to check “Add Tesseract to PATH”)

🎯 Summary

This project provides an extremely flexible and powerful OCR + AI pipeline capable of processing large batches of images, extracting meaningful text, and producing user-friendly reports.
It is designed to be robust, maintainable, and highly extensible, suitable for production use.
