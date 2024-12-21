import os
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms.c_transforms as transforms
from mindspore import nn, context, Model
from mindspore.train.callback import LossMonitor
from mindspore.nn import Accuracy
import mindspore
from mindspore.train.serialization import save_checkpoint

# GPU bağlamı ayarla
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Veri seti yükleme fonksiyonu
def create_custom_dataset(data_path, batch_size=16, repeat_size=1):
    """
    İnsan, kedi ve köpek veri setini yükler ve MindSpore formatına dönüştürür.

    Args:
        data_path (str): Veri setinin bulunduğu kök klasör.
        batch_size (int): Batch boyutu.
        repeat_size (int): Dataset'in tekrar sayısı.

    Returns:
        mindspore.dataset: Eğitim için hazırlanmış veri seti.
    """
    # Veri setini oluştur
    dataset = ds.ImageFolderDataset(data_path, class_indexing={"person": 0, "cats": 1, "dogs": 2})

    # Görüntü işlemleri
    trans = [
        vision.Decode(),  # Görüntüleri çözümle
        vision.Resize((32, 32)),  # Görüntü boyutlandır
        vision.Rescale(1.0 / 255.0, 0.0),  # Normalizasyon
        vision.HWC2CHW()  # HWC'den CHW formatına dönüştür
    ]
    type_cast_op = transforms.TypeCast(mindspore.int32)

    # Veriyi işle
    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=4)
    dataset = dataset.map(operations=type_cast_op, input_columns="label")

    # Shuffle, batch ve repeat işlemleri
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat_size)

    return dataset

# İleri Düzey CNN Modeli
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
            nn.Dense(512 * 1 * 1, 4096),  # Tam bağlantı 1
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),  # Regularization
            nn.Dense(4096, 4096),  # Tam bağlantı 2
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(4096, 3)  # Sınıflar (İnsan, Kedi, Köpek)
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

# Modeli oluştur
network = AdvancedClassificationNet()
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opt = nn.Adam(network.trainable_params(), learning_rate=0.0001)  # Daha düşük öğrenme oranı
model = Model(network, loss_fn=loss, optimizer=opt, metrics={"Accuracy": Accuracy()})

# Veri setini yükle ve eğit
data_path = "/home/eren/İndirilenler/Eren_dataset"  # Kendi veri seti yolunuzu buraya koyun
train_dataset = create_custom_dataset(data_path)

print("Model eğitimi başlıyor...")
model.train(epoch=100, train_dataset=train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=True)  # Daha fazla epoch
print("Eğitim tamamlandı!")

# Modeli kaydet
save_checkpoint(network, "advanced_classification_person_cats_dogs.ckpt")
