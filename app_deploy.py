import os
import numpy as np
import cv2
from flask import Flask, render_template, request, send_file

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


app = Flask(__name__)

CONF_THRESHOLD = 0.5


# ---------------- FAKE DETECTION ----------------
def fake_detection(image):
    h, w, _ = image.shape

    boxes = np.array([
        [int(w * 0.2), int(h * 0.2), int(w * 0.4), int(h * 0.4)],
        [int(w * 0.5), int(h * 0.3), int(w * 0.7), int(h * 0.6)],
        [int(w * 0.3), int(h * 0.5), int(w * 0.6), int(h * 0.8)]
    ])

    confidence = round(np.random.uniform(85, 95), 2)

    return boxes, confidence


# ---------------- YIELD (FAKE) ----------------
def estimate_yield(image, boxes):
    total = 0
    cluster_weights = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # fake prediction per cluster
        pred = np.random.uniform(100, 200)

        total += pred
        cluster_weights.append(float(pred))

        # draw bounding boxes
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

        # 🔥 Fake detection + yield
        boxes, confidence = fake_detection(image)
        estimated_yield, cluster_weights, image = estimate_yield(image, boxes)

        cv2.imwrite(filepath, image)

        return render_template(
            "index.html",
            filename=file.filename,
            model_used="Demo Mode",
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