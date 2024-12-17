import os
import json
import cv2

def extract_coco_categories(image_folder, annotation_file, output_folder, excluded_categories, image_size=224, max_images=10000):
    """
    Belirli kategorileri hariç tutarak diğer kategoriye ait görüntüleri COCO veri setinden çıkarır ve belirlenen klasöre kaydeder.

    Args:
        image_folder (str): Görüntülerin bulunduğu klasör yolu.
        annotation_file (str): COCO anotasyon dosyasının yolu.
        output_folder (str): Çıkarılan görüntülerin kaydedileceği klasör yolu.
        excluded_categories (list): Hariç tutulacak kategori isimlerinin listesi.
        image_size (int): Yeniden boyutlandırılacak görüntü boyutu (kare, örn: 224x224).
        max_images (int): Her kategori için işlenecek maksimum görüntü sayısı.
    """
    # Annotation dosyasını yükle
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Kategori ismi ve ID eşleştir
    category_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}
    included_categories = {cat_id: cat_name for cat_id, cat_name in category_mapping.items() if cat_name not in excluded_categories}

    print(f"İşlenecek kategoriler: {list(included_categories.values())}")

    # Görüntü ve anotasyon eşleştirmesi
    annotations = coco_data['annotations']
    images = {img['id']: img for img in coco_data['images']}

    for target_id, category_name in included_categories.items():
        print(f"İşlemde: {category_name}")

        # Hedef kategoriye ait anotasyonları filtrele
        selected_annotations = [ann for ann in annotations if ann['category_id'] == target_id]

        print(f"{category_name} kategorisi için seçilen anotasyonların sayısı: {len(selected_annotations)}")

        # Hedef kategori klasörünü oluştur
        category_folder = os.path.join(output_folder, category_name)
        os.makedirs(category_folder, exist_ok=True)

        count = 0
        for ann in selected_annotations:
            if count >= max_images:
                print(f"{category_name} için {max_images} görüntüye ulaşıldı, işlem durduruluyor.")
                break

            image_id = ann['image_id']

            # Görüntü yolunu belirle
            image_info = images[image_id]
            image_path = os.path.join(image_folder, image_info['file_name'])

            # Görüntüyü yükle ve yeniden boyutlandır
            image = cv2.imread(image_path)
            if image is None:
                print(f"Görüntü yüklenemedi: {image_path}")
                continue
            resized_image = cv2.resize(image, (image_size, image_size))

            # Görüntüyü kaydet
            output_path = os.path.join(category_folder, image_info['file_name'])
            cv2.imwrite(output_path, resized_image)
            count += 1
            print(f"{category_name} - Görüntü kaydedildi: {output_path} ({count}/{max_images})")

        print(f"Kategori {category_name} için işlem tamamlandı. Toplam {count} görüntü işlendi.")

# Kullanım
if __name__ == "__main__":
    # COCO görüntü ve anotasyon klasörleri
    image_folder = "/home/eren/İndirilenler/coco/train2017"  # Görüntülerin bulunduğu klasör
    annotation_file = "/home/eren/İndirilenler/coco/annotations_trainval2017/annotations/instances_train2017.json"  # COCO anotasyon dosyası

    # Çıkış klasörü
    output_folder = "/home/eren/İndirilenler/Eren_dataset"

    # Hariç tutulacak kategoriler
    excluded_categories = ["chair", "person", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
                           "baseball glove", "skateboard", "surfboard", "tennis racket"]

    # Fonksiyonu çalıştır
    print(f"'{output_folder}' klasörüne veriler işleniyor...")
    extract_coco_categories(image_folder, annotation_file, output_folder, excluded_categories, image_size=224, max_images=10000)
