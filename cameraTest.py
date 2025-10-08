import cv2
import numpy as np
from sklearn.cluster import KMeans

camera = cv2.VideoCapture(0)

while True:
    ok, frame_img = camera.read()
    if not ok:
        break

    img_h, img_w = frame_img.shape[:2]

    box_w, box_h = img_w // 4, img_h // 4
    x_left  = img_w // 2 - box_w // 2
    y_top   = img_h // 2 - box_h // 2
    x_right = x_left + box_w
    y_bottom = y_top + box_h

    center_roi = frame_img[y_top:y_bottom, x_left:x_right]

    pixels = center_roi.reshape(-1, 3)

    km = KMeans(n_clusters=1, n_init="auto", random_state=42)
    km.fit(pixels)
    dom_bgr = km.cluster_centers_[0].astype(np.uint8)

    cv2.rectangle(frame_img, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)

    swatch = np.zeros((100, 100, 3), dtype=np.uint8)
    swatch[:] = dom_bgr

    cv2.imshow("Live Camera", frame_img)
    cv2.imshow("Center Dominant Color", swatch)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
