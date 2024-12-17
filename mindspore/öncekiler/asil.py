import os
import numpy as np
import cv2
from mindspore import context, nn, Tensor, ops
from mindspore.train.callback import LossMonitor
from pycocotools.coco import COCO
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.nn import TrainOneStepCell, WithLossCell

# COCO sınıf adları ve ID'leri
class_names = ["person", "chair"]

# COCO sınıf ID'lerini almak için
def get_class_ids(class_names, coco):
    class_ids = []
    for name in class_names:
        cat_id = coco.getCatIds(catNms=[name])
        if cat_id:
            class_ids.append(cat_id[0])
    return class_ids

# COCO veri seti yükleyici
class COCOFilteredDataset:
    def __init__(self, image_dir, annotation_file, class_names, transform=None, max_boxes=5):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.transform = transform
        self.class_names = class_names
        self.class_ids = get_class_ids(class_names, self.coco)
        self.max_boxes = max_boxes
        self.ids = [img_id for img_id in self.coco.imgs.keys() if self.coco.getAnnIds(imgIds=img_id, catIds=self.class_ids)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.class_ids)
        annotations = self.coco.loadAnns(ann_ids)
        
        boxes = [[x, y, x + w, y + h] for ann in annotations for x, y, w, h in [ann['bbox']]]
        labels = [self.class_ids.index(ann['category_id']) for ann in annotations]

        if len(boxes) < self.max_boxes:
            boxes += [[0, 0, 0, 0]] * (self.max_boxes - len(boxes))
            labels += [0] * (self.max_boxes - len(labels))
        else:
            boxes = boxes[:self.max_boxes]
            labels = labels[:self.max_boxes]

        if self.transform is not None:
            for op in self.transform:
                image = op(image)

        return np.array(image, dtype=np.float32), np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int32)

# Veri seti yolları
data_dir = "/home/eren/İndirilenler/coco"
train_images_dir = f"{data_dir}/train2017"
train_annotation_file = "/home/eren/İndirilenler/coco/annotations_trainval2017/annotations/instances_train2017.json"

# Transform işlemleri
transform_ops = [vision.Resize((128, 128)), vision.Rescale(1.0 / 255.0, 0.0), vision.HWC2CHW()]

train_dataset = COCOFilteredDataset(
    image_dir=train_images_dir,
    annotation_file=train_annotation_file,
    class_names=class_names,
    transform=transform_ops
)
train_ds = ds.GeneratorDataset(train_dataset, ["image", "boxes", "labels"])
train_ds = train_ds.batch(1, drop_remainder=True)

# Özellik çıkarıcı katman
class FeatureExtractor(nn.Cell):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='same')
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, pad_mode='same')
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, pad_mode='same')
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, pad_mode='same')
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        return x

class RegionProposalNetwork(nn.Cell):
    def __init__(self, in_channels, mid_channels, num_anchors=9):
        super(RegionProposalNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, pad_mode='same')
        self.score = nn.Conv2d(mid_channels, num_anchors * 2, 1)
        self.loc = nn.Conv2d(mid_channels, num_anchors * 4, 1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.conv(x))
        rpn_locs = self.loc(x)
        rpn_scores = self.score(x)
        return rpn_locs, rpn_scores

class ROIHead(nn.Cell):
    def __init__(self, num_classes):
        super(ROIHead, self).__init__()
        self.fc1 = nn.Dense(512 * 7 * 7, 4096)
        self.fc2 = nn.Dense(4096, 4096)
        self.cls_loc = nn.Dense(4096, num_classes * 4)
        self.score = nn.Dense(4096, num_classes)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        roi_cls_locs = self.cls_loc(x)
        roi_scores = self.score(x)
        return roi_cls_locs, roi_scores

# Faster R-CNN modeli
class FasterRCNN(nn.Cell):
    def __init__(self, feature_extractor, rpn, roi_head):
        super(FasterRCNN, self).__init__()
        self.feature_extractor = feature_extractor
        self.rpn = rpn
        self.roi_head = roi_head

    def construct(self, x):
        features = self.feature_extractor(x)
        rpn_locs, rpn_scores = self.rpn(features)
        roi_cls_locs, roi_scores = self.roi_head(features)
        return roi_scores

feature_extractor = FeatureExtractor()
rpn = RegionProposalNetwork(in_channels=512, mid_channels=512)
roi_head = ROIHead(num_classes=2)
faster_rcnn = FasterRCNN(feature_extractor, rpn, roi_head)

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
optimizer = nn.Adam(faster_rcnn.trainable_params(), learning_rate=0.001)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# Modeli eğitmek için WithLossCell ve TrainOneStepCell kullanıyoruz
net_with_loss = WithLossCell(faster_rcnn, loss_fn)
train_net = TrainOneStepCell(net_with_loss, optimizer)
train_net.set_train()

print("Model eğitimi başlatılıyor...")
for data in train_ds.create_tuple_iterator():
    images, boxes, labels = data
    labels = labels[:, 0]  # labels'ı tek boyutlu hale getir
    loss = train_net(images, labels)
    print(f"Loss: {loss.asnumpy()}")
print("Eğitim tamamlandı!")
