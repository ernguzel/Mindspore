import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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
            nn.Dense(4096, 3)
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
param_dict = load_checkpoint("advanced_classification_person_cats_dogs.ckpt")
load_param_into_net(network, param_dict)
network.set_train(False)

# Görüntüyü işleme fonksiyonu
def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return Tensor(image[np.newaxis, ...])

# Sınıflandırıcı
def classify_image(image, threshold=0.98):
    predictions = network(image)
    probabilities = predictions.asnumpy()
    softmax = np.exp(probabilities) / np.sum(np.exp(probabilities), axis=1, keepdims=True)
    class_id = np.argmax(softmax)
    max_prob = softmax[0][class_id]

    # Eşik değerine göre karar ver
    if max_prob < threshold:
        return "Hareketsiz", softmax[0]
    return "Hareketli", softmax[0]

# ROS 2 düğümü
class ImageClassifierNode(Node):
    def __init__(self):
        super().__init__('image_classifier_node')
        self.subscription = self.create_subscription(
            Image,
            'camera/raw',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info("Image Classifier Node başlatıldı.")

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            input_image = preprocess_image(cv_image)
            label, probabilities = classify_image(input_image)

            # Sınıf oranlarını oluştur
            result_text = f"{label} ({max(probabilities):.2f})"
            for idx, prob in enumerate(probabilities):
                result_text += f" | {['Person', 'Cats', 'Dogs'][idx]}: {prob:.2f}"

            # Görüntü üzerine yaz
            cv2.putText(cv_image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("ROS 2 Camera Feed", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Görüntü işlenirken hata oluştu: {e}")

# Ana işlev
def main(args=None):
    rclpy.init(args=args)
    node = ImageClassifierNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Düğüm sonlandırılıyor.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
