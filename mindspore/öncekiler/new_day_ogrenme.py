import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import numpy as np
from mindspore import nn, context, Model
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore.train.serialization import save_checkpoint

# GPU bağlamı ayarla
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def create_coco_dataset(image_folder, annotation_file, batch_size=16, repeat_size=1, max_boxes=10):
    """
    COCO veri setini MindSpore için hazırlar ve sorunlu verileri atlar.
    """
    print("COCO veri seti yükleniyor...")
    coco_ds = ds.CocoDataset(image_folder, annotation_file=annotation_file, task="Detection")

    print("Veriler filtreleniyor...")
    def is_valid_data(image, bbox, category_id):
        """
        Geçerli bounding box ve kategori bilgisine sahip verileri döndürür.
        """
        return bbox is not None and len(bbox) > 0 and category_id is not None and len(category_id) > 0

    coco_ds = coco_ds.filter(predicate=is_valid_data, input_columns=["image", "bbox", "category_id"])

    def pad_bbox(bboxes):
        """
        Bounding box'ları sabit boyuta getirir.
        """
        if len(bboxes) > max_boxes:
            return np.array(bboxes[:max_boxes], dtype=np.float32)
        padding = np.zeros((max_boxes - len(bboxes), 4), dtype=np.float32)
        return np.vstack((bboxes, padding))

    def pad_labels(labels):
        """
        Kategori bilgilerini sabit boyuta getirir.
        """
        labels = np.array(labels, dtype=np.int32).reshape(-1, 1)
        if len(labels) > max_boxes:
            return labels[:max_boxes]
        padding = np.zeros((max_boxes - len(labels), 1), dtype=np.int32)
        return np.vstack((labels, padding))

    print("Veri işleniyor...")
    trans = [
        vision.Decode(),
        vision.Resize((32, 32)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.HWC2CHW()
    ]

    coco_ds = coco_ds.map(operations=trans, input_columns="image")
    coco_ds = coco_ds.map(operations=lambda b: pad_bbox(b), input_columns="bbox")
    coco_ds = coco_ds.map(operations=lambda l: pad_labels(l), input_columns="category_id")

    print("Sorunlu veriler kontrol ediliyor...")
    for data in coco_ds.create_dict_iterator():
        bbox_shape = data["bbox"].shape
        label_shape = data["category_id"].shape
        if bbox_shape != (max_boxes, 4) or label_shape != (max_boxes, 1):
            print(f"Sorunlu veri atlandı: bbox_shape={bbox_shape}, label_shape={label_shape}")
            continue

    print("Veri karıştırılıyor ve batch işlemi uygulanıyor...")
    coco_ds = coco_ds.shuffle(buffer_size=1000)
    coco_ds = coco_ds.batch(batch_size, drop_remainder=True)
    coco_ds = coco_ds.repeat(repeat_size)

    print("COCO veri seti hazırlandı.")
    return coco_ds

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

network = SimpleClassificationNet()
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opt = nn.Adam(network.trainable_params(), learning_rate=0.001)
model = Model(network, loss_fn=loss, optimizer=opt, metrics={"Accuracy": Accuracy()})

# COCO Veri Setini Hazırla ve Eğit
image_folder = "/home/eren/İndirilenler/coco/train2017"  # Görüntülerin bulunduğu klasör
annotation_file = "/home/eren/İndirilenler/coco/annotations_trainval2017/annotations/instances_train2017.json"  # COCO anotasyon dosyası

train_dataset = create_coco_dataset(image_folder, annotation_file)

print("Model eğitimi başlıyor...")
model.train(epoch=5, train_dataset=train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=True)
print("Eğitim tamamlandı!")

save_checkpoint(network, "motion_detection_coco.ckpt")
