FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
ENV TESTING_DIR=/data/testing
ENV MEDIAN=False
CMD ["python", "server.py"]
