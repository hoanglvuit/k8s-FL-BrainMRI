FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py model.py utils.py /app/
ENV TESTING_DIR=/data/testing
CMD ["python", "server.py"]
