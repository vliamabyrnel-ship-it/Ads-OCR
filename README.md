# Ads-OCR
robust, modular, with logging, JSON reports, and a Word document that includes each image and its extracted text

"""
ocr_ad_pipeline.py

Engineer-grade OCR pipeline for advertisement-style images.

Features
--------
* Walks a directory of images and processes each one.
* Uses EasyOCR (GPU if available, with safe CPU fallback) + optional Tesseract.
* Applies multiple image pre-processing variants (CLAHE, thresholding, etc.).
* Merges OCR results, dropping low-confidence and duplicate lines.
* Uses an AI summarizer (HuggingFace Transformers) to produce a concise
  "meaningful advertisement text" block for each image (with safe fallback).
* Writes:
    - A JSON report per image (raw lines, bounding boxes, summary).
    - A single Word (.docx) file that contains for each image:
          - The image itself
          - Raw extracted text
          - Summarized ad text

The script is structured so other engineers can maintain it easily:
    - clear data classes
    - separation between OCR, summarization, and reporting
    - logging instead of random prints
    - graceful error handling and fallbacks
"""
