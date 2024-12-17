import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_transforms

# Veri setini yükleme (data_path klasör yolunu indirilen dosyaların olduğu dizine ayarla)
data_path = "/home/eren/İndirilenler/cifar-10-batches-bin"  # Dosya yolunu kendi dizinine göre güncelle
cifar10_ds = ds.Cifar10Dataset(data_path)

# Veri setini işleme
transform_ops = [
    c_transforms.Resize((32, 32)),
    c_transforms.Rescale(1.0 / 255.0, 0.0),
    c_transforms.HWC2CHW()
]

cifar10_ds = cifar10_ds.map(operations=transform_ops, input_columns="image")
cifar10_ds = cifar10_ds.batch(32, drop_remainder=True)

# İlk veri örneğini kontrol etme
for data in cifar10_ds.create_dict_iterator():
    print(data["image"].shape, data["label"])
    break
