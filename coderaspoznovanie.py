import os, time, cv2, numpy as np
from pathlib import Path

DB_PATH     = "./db"
SFACE_ONNX  = "./face_recognition_sface_2021dec.onnx"      
YUNET_ONNX  = "./face_detection_yunet_2023mar.onnx"
THRESH      = 0.363

recognizer = cv2.FaceRecognizerSF_create(SFACE_ONNX, "")
detector   = cv2.FaceDetectorYN_create(YUNET_ONNX, "", (320,320), score_threshold=0.5, nms_threshold=0.3, top_k=5000)

def list_images(root):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            for p in d.rglob("*"):
                if p.suffix.lower() in exts:
                    yield d.name, str(p)

def embed_from_img(img):
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    ok, faces = detector.detect(img)
    out = []
    if ok and faces is not None and len(faces):
        for f in faces:
            x, y, wb, hb = f[:4].astype(int)
            row = f[:14].astype(np.float32).reshape(1, 14)  # box + 5 keypoints
            face = recognizer.alignCrop(img, row)
            feat = recognizer.feature(face)
            out.append((feat, (x, y, wb, hb)))
    return out

gallery = {}
for label, path in list_images(DB_PATH):
    img = cv2.imread(path)
    if img is None: continue
    fb = embed_from_img(img)
    if not fb: continue
    feat, _ = fb[0]                       # largest/first face per image
    gallery.setdefault(label, []).append(feat)

labels = sorted(gallery.keys())
G = []
for k in labels:
    M = np.mean(np.vstack(gallery[k]), axis=0)
    M = M / max(np.linalg.norm(M), 1e-9)
    G.append(M)
G = np.vstack(G).astype(np.float32)      

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("YuNet+SFace", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YuNet+SFace", 640, 480)
ema_fps = 0.0

while cv2.waitKey(1) != 27:
    ok, frame = cap.read()
    if not ok: break
    t0 = time.perf_counter()

    fb = embed_from_img(frame)
    for feat, (x,y,wb,hb) in fb:
        q = feat / max(np.linalg.norm(feat), 1e-9)
        sims = (G @ q.ravel()).astype(np.float32)   # cosine
        i = int(np.argmax(sims)); s = float(sims[i])

        name = labels[i] if s >= THRESH else "unknown"
        cv2.rectangle(frame, (x,y), (x+wb, y+hb), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({s:.2f})", (x, max(0,y-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    dt = max(time.perf_counter() - t0, 1e-6)
    ema_fps = (ema_fps*0.9 + 0.1*(1.0/dt)) if ema_fps else 1.0/dt
    cv2.putText(frame, f"FPS: {ema_fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.imshow("YuNet+SFace", frame)

cap.release(); cv2.destroyAllWindows()
