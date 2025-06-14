from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import traceback, io, base64, os, uuid
from datetime import datetime

app = FastAPI()
model = YOLO("best.pt")

# Ensure the folder exists
SAVE_DIR = "received"
os.makedirs(SAVE_DIR, exist_ok=True)

class DetectRequest(BaseModel):
    image_data: str

@app.post("/detect")
def detect(req: DetectRequest):
    try:
        # 1) Decode base64 payload
        img_bytes = base64.b64decode(req.image_data)

        # 2) Save to disk with a timestamp + UUID
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename  = f"{timestamp}_{uuid.uuid4().hex}.jpg"
        filepath  = os.path.join(SAVE_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        # 3) Open as PIL image for inference
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 4) Run your YOLO predict as before
        results = model.predict(source=img, conf=0.5, stream=False)

        # 5) Extract detections
        dets = []
        for r in results:
            for box in r.boxes:
                dets.append({
                    "name":       model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0])
                })
        dets.sort(key=lambda x: -x["confidence"])
        return {"top": dets[:10]}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

