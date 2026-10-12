FROM python:3.12-slim

# System libraries needed by unstructured's hi_res PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

EXPOSE 7860
CMD ["python", "app.py"]
