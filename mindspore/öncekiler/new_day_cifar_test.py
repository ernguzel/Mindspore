import cv2
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import nn

# Kamera çözünürlüğünü ayarla
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# MindSpore bağlamını ayarla
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# Modeli yükleme
class SimpleClassificationNet(nn.Cell):
    def __init__(self):
        super(SimpleClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, pad_mode='pad', padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, pad_mode='pad', padding=1)
        self.fc1 = nn.Dense(64 * 32 * 32, 128)
        self.fc2 = nn.Dense(128, 2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Eğitilmiş modeli yükle
model_path = "motion_detection.ckpt"
network = SimpleClassificationNet()
param_dict = load_checkpoint(model_path)
load_param_into_net(network, param_dict)
network.set_train(False)  # Modeli test moduna al

# Görüntü ön işleme
def preprocess_image(frame):
    """
    Kameradan alınan görüntüyü model için hazırlar.
    """
    img = cv2.resize(frame, (32, 32))  # CIFAR-10 boyutuna yeniden boyutlandır
    img = img.astype(np.float32) / 255.0  # Normalize et
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Batch boyutu ekle
    return Tensor(img)

# Kameradan canlı görüntü al ve sınıflandırma yap
cap = cv2.VideoCapture(0)  # Kamerayı başlat
cap.set(3, CAMERA_WIDTH)
cap.set(4, CAMERA_HEIGHT)

print("Kamera açık. Çıkmak için 'q' tuşuna basın.")
cv2.namedWindow("Gerçek Zamanlı Sınıflandırma")  # Tek bir pencere oluştur

while True:
    ret, frame = cap.read()  # Kameradan bir kare al
    if not ret:
        break

    # Model için görüntüyü hazırla
    input_image = preprocess_image(frame)

    # Model ile tahmin yap
    output = network(input_image)
    predicted_class = np.argmax(output.asnumpy(), axis=1)

    # Tahmini sınıfı belirle
    class_name = "Hareketli" if predicted_class == 1 else "Hareketsiz"

    # Görüntü üzerine yazdır
    cv2.putText(frame, f"Tespit: {class_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Kameradan alınan görüntüyü göster
    cv2.imshow("Gerçek Zamanlı Sınıflandırma", frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
