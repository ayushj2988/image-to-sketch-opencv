from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from sketch import image_to_sketch
import io

app = Flask(__name__)

@app.route("/sketch", methods=["POST"])
def sketch():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    sketch_img = image_to_sketch(img)

    _, buffer = cv2.imencode(".png", sketch_img)
    return send_file(
        io.BytesIO(buffer),
        mimetype="image/png",
        as_attachment=False,
        download_name="sketch.png"
    )

if __name__ == "__main__":
    app.run(debug=True)
