import os
import shutil
import random

def split_training_to_clients(training_dir, clients_dir, num_clients=3, shuffle=True):
    """
    training_dir: đường dẫn đến folder Training (chứa các class).
    clients_dir: nơi tạo các folder client_1, client_2...
    num_clients: số lượng client muốn chia dữ liệu.
    """
    # Tạo thư mục chứa client
    os.makedirs(clients_dir, exist_ok=True)

    # Danh sách class nằm trong Training
    classes = [cls for cls in os.listdir(training_dir) 
               if os.path.isdir(os.path.join(training_dir, cls))]

    print("Found classes:", classes)

    # Tạo thư mục client và các class bên trong
    for i in range(1, num_clients + 1):
        client_path = os.path.join(clients_dir, f"client_{i}")
        os.makedirs(client_path, exist_ok=True)

        for cls in classes:
            os.makedirs(os.path.join(client_path, cls), exist_ok=True)

    # Chia ảnh theo từng class
    for cls in classes:
        cls_path = os.path.join(training_dir, cls)
        images = os.listdir(cls_path)

        if shuffle:
            random.shuffle(images)

        total = len(images)
        base = total // num_clients
        extra = total % num_clients  # chia đều phần dư

        start = 0
        for i in range(num_clients):
            end = start + base + (1 if i < extra else 0)
            client_imgs = images[start:end]

            for img in client_imgs:
                src = os.path.join(cls_path, img)
                dst = os.path.join(clients_dir, f"client_{i+1}", cls, img)
                shutil.copy(src, dst)   # chỉ copy, không move

            start = end

        print(f"Class '{cls}': {total} ảnh → chia cho {num_clients} client.")

    print("\n✔ Done! Training đã được phân phối vào các client.")


# -------------------------
# Cách dùng
# -------------------------
training_path = "dataset/Training"    # Training giữ nguyên
clients_output = "dataset/Clients"    # nơi tạo client_1, client_2...
num_clients = 3                       # số client mong muốn

split_training_to_clients(training_path, clients_output, num_clients)
