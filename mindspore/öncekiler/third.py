import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms.c_transforms as transforms
from mindspore import nn, context, Model
from mindspore.train.callback import LossMonitor
from mindspore.nn import Accuracy

# GPU kullanımı için bağlamı ayarla
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# Veri setini yükle
def create_dataset(data_path, batch_size=32, repeat_size=1):
    cifar10_ds = ds.Cifar10Dataset(data_path)
    
    # Görüntü işlemleri
    trans = [
        vision.Resize((32, 32)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.HWC2CHW()
    ]
    type_cast_op = transforms.TypeCast(mindspore.int32)
    
    cifar10_ds = cifar10_ds.map(operations=trans, input_columns="image")
    cifar10_ds = cifar10_ds.map(operations=type_cast_op, input_columns="label")
    
    # Veriyi karıştır ve batch'le
    cifar10_ds = cifar10_ds.shuffle(buffer_size=1000)
    cifar10_ds = cifar10_ds.batch(batch_size, drop_remainder=True)
    cifar10_ds = cifar10_ds.repeat(repeat_size)
    
    return cifar10_ds

# Advanced CNN Modeli Tanımla
class AdvancedCNN(nn.Cell):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        
        # Convolutional katmanlar
        self.conv1 = nn.Conv2d(3, 64, 3, pad_mode='pad', padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, pad_mode='pad', padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, pad_mode='pad', padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, pad_mode='pad', padding=1)
        
        # Fully Connected (Tam Bağlantılı) Katmanlar
        self.fc1 = nn.Dense(2 * 2 * 512, 1024)
        self.fc2 = nn.Dense(1024, 512)
        self.fc3 = nn.Dense(512, 10)
        
        # Diğer katmanlar
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Overfitting'i önlemek için

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

# Modeli oluştur
network = AdvancedCNN()
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opt = nn.Adam(network.trainable_params(), learning_rate=0.001)
model = Model(network, loss_fn=loss, optimizer=opt, metrics={"Accuracy": Accuracy()})

# Veri setini hazırla
data_path = "/home/eren/İndirilenler/cifar-10-batches-bin"  # CIFAR-10 veri seti konumu
train_dataset = create_dataset(data_path)

# Modeli eğit
print("Advanced CNN modeli ile eğitim başlıyor...")
model.train(epoch=5, train_dataset=train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=True)
print("Eğitim tamamlandı!")

# Modeli kaydet
from mindspore.train.serialization import save_checkpoint
save_checkpoint(network, "cifar10_advanced_cnn.ckpt")

