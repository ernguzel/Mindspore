import cv2
import numpy as np
from mindspore import Tensor, dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.dataset.vision import Rescale, Resize, HWC2CHW
from mindspore import nn

# Model Tanımı
class AdvancedClassificationNet(nn.Cell):
    def __init__(self):
        super(AdvancedClassificationNet, self).__init__()
        # İlk konvolüsyon bloğu
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(3, 64, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        # İkinci konvolüsyon bloğu
        self.conv2 = nn.SequentialCell([
            nn.Conv2d(64, 128, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        # Üçüncü konvolüsyon bloğu
        self.conv3 = nn.SequentialCell([
            nn.Conv2d(128, 256, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        # Dördüncü konvolüsyon bloğu
        self.conv4 = nn.SequentialCell([
            nn.Conv2d(256, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        # Beşinci konvolüsyon bloğu
        self.conv5 = nn.SequentialCell([
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        # Tam bağlantı katmanları
        self.fc = nn.SequentialCell([
            nn.Dense(512, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(4096, 7)  # Sınıf sayısını 7 olarak ayarladık
        ])

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc(x)
        return x

# Görüntü ön işleme
def preprocess_frame(frame):
    transform = [
        Resize((32, 32)),
        Rescale(1.0 / 255.0, 0.0),
        HWC2CHW()
    ]
    for op in transform:
        frame = op(frame)
    return frame

# Sınıf tahmini
def predict_class(frame, model):
    frame_tensor = Tensor(np.expand_dims(frame, axis=0), dtype=mstype.float32)
    logits = model(frame_tensor)
    probabilities = nn.Softmax()(logits).asnumpy().flatten()
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class] * 100
    return predicted_class, confidence

# Modeli yükle
model = AdvancedClassificationNet()
param_dict = load_checkpoint("/home/eren/Eren/projects/Mindspore/mindspore/advanced_motion_detection_eren_dataset.ckpt")
load_param_into_net(model, param_dict)
model.set_train(False)

# Sınıf etiketleri
class_labels = ["person", "chair", "keyboard", "laptop", "bottle", "dog", "bicycle"]

# Kamerayı başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

print("Kamera açık. Çıkmak için 'q' tuşuna basın.")
cv2.namedWindow("Gerçek Zamanlı Sınıflandırma")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    try:
        # Görüntüyü işleyip sınıflandır
        processed_frame = preprocess_frame(frame)
        predicted_class, confidence = predict_class(processed_frame, model)

        # Sınıf etiketini göster
        label = f"{class_labels[predicted_class]}: {confidence:.2f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gerçek Zamanlı Sınıflandırma", frame)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
