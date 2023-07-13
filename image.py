import cv2
import os

# Görüntünün bulunduğu dizini belirtin
image_directory = "C:\\Users\\erena\\OneDrive\Desktop\\Data Mining"
image_filename = "body.png"

# Görüntünün tam yolunu oluşturun
image_path = os.path.join(image_directory, image_filename)

# Görüntüyü yükle
image = cv2.imread(image_path)

if image is not None:
    # Önceden eğitilmiş insan algılama modelini yükle
    human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    # Görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # İnsanları algıla
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=4, minSize=(30, 30))

    # Algılanan insan sayısını yazdır
    print("Algılanan insan sayısı:", len(humans))

    # Algılanan insanları görselleştir
    for (x, y, w, h) in humans:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Sonucu göster
    cv2.imshow("Sonuç", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Görüntü yüklenemedi!")
