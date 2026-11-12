FROM python:3.11-slim

WORKDIR /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install pandas openpyxl sentence_transformers datasets uvicorn fastapi pydantic matplotlib seaborn accelerate>=1.1.0

EXPOSE 8000

COPY . .

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]