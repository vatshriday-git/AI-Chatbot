FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Pre-download models & NLTK data at build time
RUN python -c "from nlp_engine import ChatEngine; ChatEngine()"
ENV ADMIN_TOKEN=change-me
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
