import cv2
import numpy as np

cam = cv2.VideoCapture(0)

R1_LOW, R1_HIGH = np.array([0,   70, 70], np.uint8), np.array([10, 255, 255], np.uint8)
R2_LOW, R2_HIGH = np.array([170, 70, 70], np.uint8), np.array([179, 255, 255], np.uint8)
G_LOW,  G_HIGH  = np.array([35,  70, 70], np.uint8), np.array([85,  255, 255], np.uint8)
B_LOW,  B_HIGH  = np.array([100, 70, 70], np.uint8), np.array([130, 255, 255], np.uint8)

color_mode = "g" 
min_area = 1000
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ok, frame = cam.read()
    if not ok:
        break

    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    if color_mode == "r":
        m1 = cv2.inRange(hsv, R1_LOW, R1_HIGH)
        m2 = cv2.inRange(hsv, R2_LOW, R2_HIGH)
        mask = cv2.bitwise_or(m1, m2)
        label = "RED"
    elif color_mode == "g":
        mask = cv2.inRange(hsv, G_LOW, G_HIGH)
        label = "GREEN"
    else:
        mask = cv2.inRange(hsv, B_LOW, B_HIGH)
        label = "BLUE"

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        if a > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}  Area:{int(a)}", (x, max(0, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f"Target: {label}  (press r/g/b)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,220), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Tracked Object", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k in (ord('r'), ord('g'), ord('b')):
        color_mode = chr(k)

cam.release()
cv2.destroyAllWindows()
