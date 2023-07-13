import cv2

# Video dosyasını yükle
video = cv2.VideoCapture(r"C:\Users\erena\OneDrive\Desktop\Data Mining\humanvideo.mp4")

# Önceden eğitilmiş insan algılama modelini yükle
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

frame_count = 0
human_count = 0

while True:
    # Video akışından bir kare al
    ret, frame = video.read()
    
    # Videonun sonuna gelindiğinde veya okuma hatası olduğunda döngüden çık
    if not ret:
        break
    
    # Görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # İnsanları algıla
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Algılanan insanları say ve görselleştir
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        human_count += 1
    
    # Kareyi ekranda göster
    cv2.imshow("Video", frame)
    
    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Kare sayısını güncelle
    frame_count += 1

# Sonuçları yazdır
print("Toplam kare sayısı:", frame_count)
print("Algılanan insan sayısı:", human_count)

# Video ve pencereleri serbest bırak
video.release()
cv2.destroyAllWindows()
