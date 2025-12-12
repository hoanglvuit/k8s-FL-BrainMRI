FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY client.py model.py utils.py /app/
ENV SERVER_ADDRESS=server:8080
ENV CLIENT_DATA_DIR=/data/client_1
CMD ["python", "client.py"]
