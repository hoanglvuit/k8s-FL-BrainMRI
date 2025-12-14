# Federated Learning for Brain Tumor MRI Classification

Dự án nghiên cứu và triển khai Federated Learning (Học liên kết) cho bài toán phân loại u não từ ảnh MRI sử dụng Kubernetes để mô phỏng môi trường server và các bệnh viện (hospitals).

## 📋 Tổng quan

Dự án này triển khai một hệ thống Federated Learning sử dụng framework Flower (FLwr) để huấn luyện mô hình CNN phân loại u não từ ảnh MRI. Hệ thống được triển khai trên Kubernetes (k8s) với Minikube để mô phỏng:
- **Server**: Trung tâm điều phối quá trình huấn luyện và tổng hợp mô hình
- **Clients**: 3 bệnh viện (hospitals) độc lập, mỗi bệnh viện có dataset riêng và không chia sẻ dữ liệu

## 🎯 Mục tiêu

- Triển khai Federated Learning cho bài toán phân loại ảnh y tế
- Bảo vệ quyền riêng tư dữ liệu (dữ liệu không rời khỏi bệnh viện)
- So sánh hiệu năng giữa Federated Learning và Centralized Learning

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐
│  FL Server      │  ← Tổng hợp mô hình từ các clients
│  (K8s Pod)      │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │         │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Client │ │Client │ │Client │ │ ...   │
│  1    │ │  2    │ │  3    │ │       │
│(Hosp) │ │(Hosp) │ │(Hosp) │ │       │
└───────┘ └───────┘ └───────┘ └───────┘
```

## 📁 Cấu trúc thư mục

```
NT533-federated_learning/
├── client/                    # Client code (bệnh viện)
│   ├── client.py             # Flower client implementation
│   ├── model.py              # CNN model definition
│   ├── utils.py              # Utility functions
│   ├── Dockerfile            # Docker image cho client
│   └── requirements.txt      # Python dependencies
├── server/                    # Server code (trung tâm)
│   ├── server.py             # Flower server implementation
│   ├── model.py              # CNN model definition
│   ├── utils.py              # Utility functions
│   ├── Dockerfile            # Docker image cho server
│   └── requirements.txt      # Python dependencies
├── k8s/                       # Kubernetes deployment files
│   ├── server-deployment.yaml
│   ├── service-server.yaml
│   ├── client-deployment-1.yaml
│   ├── client-deployment-2.yaml
│   └── client-deployment-3.yaml
├── centralized-training/      # Code cho centralized training (so sánh)
│   ├── main.py
│   ├── model.py
│   └── utils.py
├── dataset/                   # Dataset MRI
│   ├── Testing/              # Testing data
│   └── Clients/              # Dữ liệu đã chia cho từng client
│       ├── client_1/
│       ├── client_2/
│       └── client_3/
└── split_dataset.py          # Script chia dataset cho các clients
```

## 🚀 Hướng dẫn triển khai

### Yêu cầu hệ thống

- Docker
- Kubernetes (Minikube)
- Python 3.8+

### Bước 1: Khởi động Minikube cluster

```powershell
minikube start --driver=docker --cpus=4 --memory=4048
```

### Bước 2: Mount dataset vào Minikube

```powershell
minikube mount "F:/Máy tính/NT533-federated_learning/dataset:/data/dataset"
```

Lưu ý: Giữ terminal này chạy trong suốt quá trình thực nghiệm.

### Bước 3: Cấu hình Docker environment để sử dụng Docker daemon của Minikube

Mở terminal PowerShell mới và chạy:

```powershell
minikube -p minikube docker-env | Invoke-Expression
```

### Bước 4: Chia dataset cho các clients (nếu chưa có)

```powershell
python split_dataset.py
```

Script này sẽ chia dữ liệu từ `dataset/Training/` thành 3 phần cho `client_1`, `client_2`, và `client_3`.

### Bước 5: Build Docker images

**Build client image:**
```powershell
cd client
docker build -t fl-client:latest .
cd ..
```

**Build server image:**
```powershell
cd server
docker build -t fl-server:latest .
cd ..
```

### Bước 6: Deploy các services và pods lên Kubernetes

```powershell
# Deploy server service
kubectl apply -f k8s/service-server.yaml

# Deploy server pod
kubectl apply -f k8s/server-deployment.yaml

# Deploy các client pods
kubectl apply -f k8s/client-deployment-1.yaml
kubectl apply -f k8s/client-deployment-2.yaml
kubectl apply -f k8s/client-deployment-3.yaml
```

### Bước 7: Theo dõi quá trình huấn luyện

**Xem logs của server:**
```powershell
kubectl logs -f deploy/fl-server
```

**Xem logs của client:**
```powershell
kubectl logs -f deploy/fl-client-1
kubectl logs -f deploy/fl-client-2
kubectl logs -f deploy/fl-client-3
```

### Bước 8: Kiểm tra trạng thái pods

```powershell
kubectl get pods
kubectl get services
```

## 📊 Kết quả thực nghiệm

### So sánh Federated Learning vs Centralized Learning

| Metric | Federated Learning | Centralized Learning | Random Model |
|--------|-------------------|---------------------|--------------|
| **Accuracy** | 81.55% | 82.77% | 29.36% |
| **F1-score** | 78.4% | 79.8% | 13.28% |

### Nhận xét

- **Federated Learning đạt hiệu năng gần bằng Centralized Learning** (chênh lệch chỉ ~1.2% về accuracy)
- Federated Learning bảo vệ được quyền riêng tư dữ liệu (dữ liệu không rời khỏi các bệnh viện)
- Random Model cho kết quả rất thấp, chứng tỏ mô hình đã học được các đặc trưng có ý nghĩa

## 🔧 Cấu hình

### Server Configuration
- Port: 8080
- Strategy: FedAvg (Federated Averaging)
- Min clients: 3
- Evaluation: Sử dụng test dataset sau mỗi round

### Client Configuration
- Local epochs: 2
- Batch size: 4
- Optimizer: SGD với learning rate 0.01
- Loss function: CrossEntropyLoss

### Model Architecture

SimpleCNN với cấu trúc:
- Conv2d(3 → 8 channels) + ReLU + MaxPool
- Conv2d(8 → 16 channels) + ReLU + MaxPool
- Fully Connected (16×56×56 → 64) + ReLU
- Fully Connected (64 → num_classes)

## 🧪 So sánh với Centralized Training

Để chạy centralized training để so sánh:

```powershell
cd centralized-training
python main.py
```

## 📚 Thư viện sử dụng

- **Flower (FLwr)**: Framework cho Federated Learning
- **PyTorch**: Deep learning framework
- **Kubernetes**: Container orchestration
- **Docker**: Containerization

## 🔍 Troubleshooting

### Vấn đề: Pods không khởi động được

- Kiểm tra images đã được build đúng chưa: `docker images | grep fl-`
- Kiểm tra logs: `kubectl describe pod <pod-name>`

### Vấn đề: Client không kết nối được với Server

- Kiểm tra service đã được tạo: `kubectl get svc`
- Kiểm tra SERVER_ADDRESS trong client deployment

### Vấn đề: Không mount được dataset

- Đảm bảo terminal mount đang chạy
- Kiểm tra đường dẫn dataset trong mount command

## 📝 Ghi chú

- Dataset phải được chia thành các thư mục class (ví dụ: glioma, meningioma, pituitary, no_tumor)
- Mỗi client chỉ có quyền truy cập dữ liệu của mình
- Server chỉ nhận model weights từ clients, không nhận dữ liệu thô

## 👥 Tác giả

Dự án nghiên cứu NT533 - Federated Learning for Brain Tumor MRI Classification

## 📄 License

[MIT License] (hoặc license phù hợp với dự án của bạn)

