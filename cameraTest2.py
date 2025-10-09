""" This is a script which gets the dominant color using OpenCV,
Sources/References: https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html 
Tutorial: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
KMeans: https://github.com/opencv/opencv/blob/master/samples/python/kmeans.py
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

def to_hex(bgr):
    b, g, r = (int(x) for x in bgr)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def make_bar(weights, centers_bgr, width=320, height=50):
    order = np.argsort(weights)[::-1]
    bar = np.zeros((height, width, 3), np.uint8)
    run = 0
    for idx in order:
        w = int(round(weights[idx] * width))
        end = min(run + w, width)
        cv2.rectangle(bar, (run, 0), (end, height - 1),
                      tuple(int(v) for v in centers_bgr[idx]), -1)
        run = end
    return bar

def dominant_from_roi(roi_bgr, k=3, subsample=4, max_pts=5000, use_lab=True):
    if subsample > 1:
        roi_bgr = roi_bgr[::subsample, ::subsample]
    space = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB) if use_lab else roi_bgr
    pts = space.reshape(-1, 3)
    n = pts.shape[0]
    if n > max_pts:
        sel = np.random.choice(n, max_pts, replace=False)
        pts = pts[sel]
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(pts)
    centers = km.cluster_centers_
    counts = np.bincount(labels, minlength=k).astype(float)
    weights = counts / counts.sum()
    dom_idx = int(np.argmax(weights))
    if use_lab:
        centers_u8 = np.uint8(centers.reshape(-1, 1, 3))
        centers_bgr = cv2.cvtColor(centers_u8, cv2.COLOR_Lab2BGR).reshape(-1, 3)
    else:
        centers_bgr = centers
    return centers_bgr[dom_idx], weights, centers_bgr

def main():
    cam = cv2.VideoCapture(0)  
    if not cam.isOpened():
        raise SystemExit("Camera not available. Try a different index or backend.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    k = 3
    subs = 4
    use_lab = True

    while True:
        ok, frame_bgr = cam.read()
        if not ok:
            break

        H, W = frame_bgr.shape[:2]
        rW, rH = 160, 160
        x0, y0 = W // 2 - rW // 2, H // 2 - rH // 2
        x1, y1 = x0 + rW, y0 + rH
        roi_bgr = frame_bgr[y0:y1, x0:x1]

        dom_bgr, mix, ctrs_bgr = dominant_from_roi(
            roi_bgr, k=k, subsample=subs, max_pts=5000, use_lab=use_lab
        )

        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)

        patch = np.zeros((100, 100, 3), np.uint8); patch[:] = dom_bgr
        bar = make_bar(mix, ctrs_bgr, width=320, height=50)

        txt = "Dominant: {} {}".format(tuple(int(c) for c in dom_bgr), to_hex(dom_bgr))
        cv2.putText(frame_bgr, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 220, 25), 2)

        # ===== ADDED: separate window that shows only the dominant color swatch + label =====
        swatch_h, swatch_w = 240, 240
        swatch = np.zeros((swatch_h, swatch_w, 3), np.uint8)
        swatch[:] = dom_bgr

        rgb = (int(dom_bgr[2]), int(dom_bgr[1]), int(dom_bgr[0]))  # BGR -> RGB
        hexcode = to_hex(dom_bgr)
        luminance = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        txt_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

        cv2.putText(swatch, f"{rgb}  {hexcode}", (10, swatch_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2, cv2.LINE_AA)
        cv2.imshow("Dominant Color", swatch)
        # ================================================================================

        cv2.imshow("Webcam", frame_bgr)
        cv2.imshow("Dominant (swatch)", patch)
        cv2.imshow("Mixture (bar)", bar)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
