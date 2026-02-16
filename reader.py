from ultralytics import YOLO
import cv2
import time 

model = YOLO("shdfinal.pt") 

cap = cv2.VideoCapture(0)
GUVEN_ESIGI = 0.50

onceki_zaman = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    su_anki_zaman = time.time()
    if su_anki_zaman != onceki_zaman:
        fps = 1 / (su_anki_zaman - onceki_zaman)
    else:
        fps = 0
    onceki_zaman = su_anki_zaman

    results = model(frame, conf=GUVEN_ESIGI, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            sinif_adi = model.names[cls_id]
            if "no" in sinif_adi.lower(): 
                renk = (0, 0, 255)
            else:
                renk = (0, 255, 0)

            durum_metni = f"{sinif_adi} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 3)
            (w, h), _ = cv2.getTextSize(durum_metni, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), renk, -1)
            cv2.putText(frame, durum_metni, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Kalite Kontrol (Orjinal Etiket)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()