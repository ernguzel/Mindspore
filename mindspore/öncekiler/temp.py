# COCO Veri Setini Hazırla ve Eğit
image_folder = "/home/eren/İndirilenler/coco/train2017"  # Görüntülerin bulunduğu klasör
annotation_file = "/home/eren/İndirilenler/coco/annotations_trainval2017/annotations/instances_train2017.json"  # COCO anotasyon dosyası
from mindspore.train.serialization import save_checkpoint