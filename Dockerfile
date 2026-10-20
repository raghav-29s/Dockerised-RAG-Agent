FROM python:3.11-slim
WORKDIR /app
# CPU-only torch: the default PyPI wheel pulls ~2 GB of CUDA libraries this app never uses.
# Its own layer, so editing requirements.txt doesn't re-download it.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
