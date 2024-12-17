import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms.c_transforms as transforms
from mindspore import nn, context, Model
from mindspore.train.callback import LossMonitor
from mindspore.nn import Accuracy
import numpy as np
import mindspore

# GPU kullanımı için bağlamı ayarla
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# Hareketli ve Hareketsiz Etiketlerini Grupla
def cifar10_label_grouping(label):
    """
    CIFAR-10 etiketlerini 'hareketli' (1) ve 'hareketsiz' (0) olarak gruplandır.
    """
    hareketli = [1, 2, 3, 5]  # araba, kuş, kedi, köpek
    hareketsiz = [0, 8, 9]    # uçak, gemi, sandalye
    return 1 if label in hareketli else 0  # Direkt int döndür

# CIFAR-10 Veri Setini Yükle
def create_cifar10_dataset(data_path, batch_size=16, repeat_size=1):
    cifar10_ds = ds.Cifar10Dataset(data_path)

    # Görüntü işlemleri
    trans = [
        vision.Resize((32, 32)),  # Görüntü boyutlandır
        vision.Rescale(1.0 / 255.0, 0.0),  # Normalizasyon
        vision.HWC2CHW()  # HWC'den CHW formatına dönüştür
    ]
    type_cast_op = transforms.TypeCast(mindspore.int32)

    # Etiketleri 'hareketli' ve 'hareketsiz' olarak değiştir
    cifar10_ds = cifar10_ds.map(operations=lambda x: cifar10_label_grouping(x), input_columns="label")
    cifar10_ds = cifar10_ds.map(operations=trans, input_columns="image")
    cifar10_ds = cifar10_ds.map(operations=type_cast_op, input_columns="label")

    # Shuffle, batch ve repeat işlemleri
    cifar10_ds = cifar10_ds.shuffle(buffer_size=1000)
    cifar10_ds = cifar10_ds.batch(batch_size, drop_remainder=True)
    cifar10_ds = cifar10_ds.repeat(repeat_size)

    return cifar10_ds

# Basit Bir CNN Modeli Tanımla
class SimpleClassificationNet(nn.Cell):
    def __init__(self):
        super(SimpleClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, pad_mode='pad', padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, pad_mode='pad', padding=1)
        self.fc1 = nn.Dense(64 * 32 * 32, 128)
        self.fc2 = nn.Dense(128, 2)  # Hareketli ve Hareketsiz için 2 sınıf

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

# Modeli Oluştur
network = SimpleClassificationNet()
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opt = nn.Adam(network.trainable_params(), learning_rate=0.001)
model = Model(network, loss_fn=loss, optimizer=opt, metrics={"Accuracy": Accuracy()})

# Veri Setini Yükle ve Eğit
data_path = "/home/eren/İndirilenler/cifar-10-batches-bin"  # CIFAR-10 veri setinin yolu
train_dataset = create_cifar10_dataset(data_path)

print("Model eğitimi başlıyor...")
model.train(epoch=5, train_dataset=train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=True)
print("Eğitim tamamlandı!")

# Modeli Kaydet
from mindspore.train.serialization import save_checkpoint
save_checkpoint(network, "motion_detection.ckpt")
