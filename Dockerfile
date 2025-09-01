FROM python:3.11-slim

# Fonts so ReportLab PDFs render nicely
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Chainlit should write files here in Spaces
ENV PERSIST_DIR=/data
RUN mkdir -p /data

# HF provides $PORT at runtime; bind to 0.0.0.0
EXPOSE 7860
CMD ["bash", "-lc", "chainlit run app.py --host 0.0.0.0 --port ${PORT:-7860}"]
