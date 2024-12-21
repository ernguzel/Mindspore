import cv2
import numpy as np
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor, context
from mindspore import nn

# MindSpore bağlamını CPU'ya ayarla
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# AdvancedClassificationNet sınıfı
class AdvancedClassificationNet(nn.Cell):
    def __init__(self):
        super(AdvancedClassificationNet, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(3, 64, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.conv2 = nn.SequentialCell([
            nn.Conv2d(64, 128, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.conv3 = nn.SequentialCell([
            nn.Conv2d(128, 256, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.conv4 = nn.SequentialCell([
            nn.Conv2d(256, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.conv5 = nn.SequentialCell([
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.fc = nn.SequentialCell([
            nn.Dense(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(4096, 3)  # Sadece "hareketli" ve "hareketsiz"
        ])

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

# Modeli yükle
network = AdvancedClassificationNet()
param_dict = load_checkpoint("move_or_not_move_detection.ckpt")
load_param_into_net(network, param_dict)
network.set_train(False)

# Görüntüyü işleme fonksiyonu
def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return Tensor(image[np.newaxis, ...])

# Sınıflandırıcı
def classify_image(image):
    predictions = network(image)
    probabilities = predictions.asnumpy()
    softmax = np.exp(probabilities) / np.sum(np.exp(probabilities), axis=1, keepdims=True)
    class_id = np.argmax(softmax)
    return ("Hareketli" if class_id == 0 else "Hareketsiz", softmax[0])

# Kamera ile görüntü yakalama
cap = cv2.VideoCapture(0)

print("Kamera açıldı, 'q' ile çıkış yapabilirsiniz.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    input_image = preprocess_image(frame)
    label, probabilities = classify_image(input_image)

    # Sınıf oranlarını oluştur
    result_text = f"{label} ({max(probabilities):.2f})"
    cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
