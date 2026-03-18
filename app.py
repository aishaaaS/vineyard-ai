import os
import numpy as np
import cv2
from flask import Flask, render_template, request, send_file

# 🔥 Detect if running on Render
DEPLOY = os.environ.get("RENDER") == "true"

# Only import heavy libs locally
if not DEPLOY:
    import torch
    import tensorflow as tf
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from ultralytics import YOLO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# ---------------- LOCAL MODEL LOAD ----------------
if not DEPLOY:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    faster_model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = faster_model.roi_heads.box_predictor.cls_score.in_features
    faster_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    faster_model.load_state_dict(torch.load("models/FASTER_RCNN.pth", map_location=device))
    faster_model.to(device)
    faster_model.eval()

    yolo_model = YOLO("models/yolov8.pt")
    minn_model = tf.keras.models.load_model("models/minn_final.keras")

CONF_THRESHOLD = 0.5


# ---------------- DETECTION ----------------
def detect_faster(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(device)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = faster_model(img_tensor)

    boxes = prediction[0]["boxes"].cpu().numpy()
    scores = prediction[0]["scores"].cpu().numpy()
    mask = scores > CONF_THRESHOLD

    return boxes[mask], round(
        float(np.mean(scores[mask])) * 100 if len(scores[mask]) > 0 else 0, 2
    )


def detect_yolo(image):
    results = yolo_model(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    mask = scores > CONF_THRESHOLD

    return boxes[mask], round(
        float(np.mean(scores[mask])) * 100 if len(scores[mask]) > 0 else 0, 2
    )


# ---------------- YIELD ----------------
def estimate_yield(image, boxes):
    total = 0
    cluster_weights = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (128, 128))
        crop = crop / 255.0
        crop = np.expand_dims(crop, axis=0)

        area = (x2 - x1) * (y2 - y1)
        area_input = np.array([[area]])

        pred = minn_model.predict([crop, area_input], verbose=0)[0][0]

        total += pred
        cluster_weights.append(float(pred))

        cv2.rectangle(image, (x1, y1), (x2, y2), (34, 139, 34), 2)

    return float(round(total / 1000, 2)), cluster_weights, image


# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            return "No file selected", 400

        upload_path = "static/uploads"
        os.makedirs(upload_path, exist_ok=True)

        filepath = os.path.join(upload_path, file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)

        # 🚀 DEPLOY MODE (NO MODELS)
        if DEPLOY:
            return render_template(
                "index.html",
                filename=file.filename,
                model_used="Demo Mode",
                detected_clusters=5,
                estimated_yield=2.4,
                confidence=92.5,
                cluster_weights=[120, 150, 130, 140, 160]
            )

        # 🔥 LOCAL MODE (REAL MODELS)
        model_choice = request.form.get("model")

        if model_choice == "faster":
            boxes, confidence = detect_faster(image)
            model_used = "Faster R-CNN"

        elif model_choice == "yolo":
            boxes, confidence = detect_yolo(image)
            model_used = "YOLOv8"

        else:
            boxes, confidence = detect_faster(image)
            model_used = "MINN"

        estimated_yield, cluster_weights, image = estimate_yield(image, boxes)

        cv2.imwrite(filepath, image)

        return render_template(
            "index.html",
            filename=file.filename,
            model_used=model_used,
            detected_clusters=len(boxes),
            estimated_yield=estimated_yield,
            confidence=confidence,
            cluster_weights=cluster_weights
        )

    return render_template("index.html")


# ---------------- PDF ----------------
@app.route("/download_report/<yield_value>")
def download_report(yield_value):

    file_path = "static/yield_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Vineyard Yield Estimation Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Estimated Yield: {yield_value} kg", styles["Normal"]))

    doc.build(elements)

    return send_file(file_path, as_attachment=True)


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)