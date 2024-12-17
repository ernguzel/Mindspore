import cv2
import numpy as np
from mindspore import Tensor
import mindspore.dataset.vision as vision
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Model
import mindspore.nn as nn

# CIFAR-10 sınıf etiketleri
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# Advanced CNN Modeli Tanımla (Önceden tanımladığımız modeli burada tekrar tanımlamamız gerekiyor)
class AdvancedCNN(nn.Cell):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, pad_mode='pad', padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, pad_mode='pad', padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, pad_mode='pad', padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, pad_mode='pad', padding=1)
        self.fc1 = nn.Dense(2 * 2 * 512, 1024)
        self.fc2 = nn.Dense(1024, 512)
        self.fc3 = nn.Dense(512, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Modeli oluştur ve parametreleri yükle
network = AdvancedCNN()
load_checkpoint("cifar10_advanced_cnn.ckpt", net=network)  # Model parametrelerini yükleyin
model = Model(network)  # Modeli MindSpore Model API'si ile sarın

# OpenCV kullanarak kamerayı başlat
cap = cv2.VideoCapture(0)

# Görüntüyü işleme fonksiyonu
def preprocess_image(frame):
    transform_ops = [
        vision.Resize((32, 32)),  # CIFAR-10 boyutuna küçült
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.HWC2CHW()
    ]
    for op in transform_ops:
        frame = op(frame)
    frame = frame.astype(np.float32)
    return frame

print("Kameradan nesne tanıma başlatıldı. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü CIFAR-10 formatına dönüştür
    processed_frame = preprocess_image(frame)
    input_tensor = Tensor(np.expand_dims(processed_frame, axis=0))

    # Model ile tahmin yap
    output = model.predict(input_tensor)
    predicted_class = np.argmax(output.asnumpy(), axis=1)[0]
    label = class_names[predicted_class]

    # Ekranda tahmini göster
    cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Nesneyi bir dikdörtgen içine almak
    height, width, _ = frame.shape
    start_point = (int(width * 0.3), int(height * 0.3))  # Dikdörtgenin başlangıç noktası
    end_point = (int(width * 0.7), int(height * 0.7))    # Dikdörtgenin bitiş noktası
    color = (0, 255, 0)  # Dikdörtgen rengi
    thickness = 2  # Çizgi kalınlığı
    cv2.rectangle(frame, start_point, end_point, color, thickness)

    cv2.imshow("Real-Time CIFAR-10 Recognition", frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
