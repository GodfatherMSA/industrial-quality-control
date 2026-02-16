from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("shdfinal.pt")

cap_ust = cv2.VideoCapture(0)
cap_yan = cv2.VideoCapture(1)

font = cv2.FONT_HERSHEY_SIMPLEX

eslesme_kurallari = {
    "bogurtlen_top": "bogurtlen_side",
    "visne_top": "visne_side",
    "frambuaz_top": "frambuaz_side",
    "canlandiran_meyveler_top": "canlandiran_meyveler_side",
    "yabanmersin_top": "yabanmersin_side",
    "nar_top": "nar_side",
    "cilek_top": "cilek_side",
    "4luormanmeyve_top": "4luormanmeyve_side",
    "3luormanmeyve_top": "3luormanmeyve_side",
}

while True:
    ret1, frame_ust = cap_ust.read()
    ret2, frame_yan = cap_yan.read()

    if not ret1 or not ret2:
        break

    frame_yan = cv2.resize(frame_yan, (640, 480))
    frame_ust = cv2.resize(frame_ust, (640, 480))

    tespit_kapak = None
    tespit_yan = None
    tarih_var_mi = False
    tarih_etiketi = ""      

    results_ust = model(frame_ust, conf=0.50, verbose=False)
    
    for r in results_ust:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            
            if "expdate" in label:
                if "no" in label:
                    tarih_var_mi = False
                    tarih_etiketi = "TARIH YOK"
                    renk = (0, 0, 255)
                else:
                    tarih_var_mi = True
                    tarih_etiketi = "TARIH VAR"
                    renk = (0, 255, 0)

                cv2.rectangle(frame_ust, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(frame_ust, tarih_etiketi, (x1, y1-10), font, 0.6, renk, 2)

            elif label in eslesme_kurallari:
                tespit_kapak = label
                cv2.rectangle(frame_ust, (x1, y1), (x2, y2), (255, 100, 0), 2) # Mavi
                cv2.putText(frame_ust, label, (x1, y1-10), font, 0.6, (255, 100, 0), 2)


    results_yan = model(frame_yan, conf=0.50, verbose=False)

    for r in results_yan:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]

            if label in eslesme_kurallari.values():
                tespit_yan = label
                cv2.rectangle(frame_yan, (x1, y1), (x2, y2), (0, 255, 255), 2) # Sarı
                cv2.putText(frame_yan, label, (x1, y1-10), font, 0.6, (0, 255, 255), 2)
    
    mesaj = "URUN ARANIYOR..."
    renk_panel = (50, 50, 50)
    renk_yazi = (255, 255, 255)

    if tespit_kapak and tespit_yan:
        
        olmasi_gereken = eslesme_kurallari.get(tespit_kapak)
        
        if tespit_yan == olmasi_gereken:
            
            if tarih_var_mi:
            
                mesaj = f"ONAYLANDI: {tespit_kapak} (TARIH OK)"
                renk_panel = (0, 255, 0)
                renk_yazi = (0, 0, 0)
            else:
                mesaj = "HATA: URUN DOGRU AMA TARIH YOK!"
                renk_panel = (0, 0, 255)
                renk_yazi = (255, 255, 0)
        else:
            mesaj = f"HATA: URUNLER FARKLI! ({tespit_kapak} != {tespit_yan})"
            renk_panel = (0, 0, 255)
            renk_yazi = (255, 255, 255)
            
    elif tarih_etiketi == "TARIH YOK":
         mesaj = "HATA: TARIH BASILMAMIS!"
         renk_panel = (0, 0, 255)
         renk_yazi = (255, 255, 0)

    birlesik = np.hstack((frame_ust, frame_yan))
    
    panel = np.zeros((100, birlesik.shape[1], 3), dtype=np.uint8)
    panel[:] = renk_panel
    
    font_scale = 1.0
    kalinlik = 2
    (text_w, text_h), _ = cv2.getTextSize(mesaj, font, font_scale, kalinlik)
    
    if text_w > birlesik.shape[1]:
        font_scale = 0.8
        (text_w, text_h), _ = cv2.getTextSize(mesaj, font, font_scale, kalinlik)
        
    text_x = int((birlesik.shape[1] - text_w) / 2)
    text_y = int((100 + text_h) / 2)
    
    cv2.putText(panel, mesaj, (text_x, text_y), font, font_scale, renk_yazi, kalinlik)

    final = np.vstack((panel, birlesik))
    cv2.imshow("Kalite Kontrol", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_ust.release()
cap_yan.release()
cv2.destroyAllWindows()