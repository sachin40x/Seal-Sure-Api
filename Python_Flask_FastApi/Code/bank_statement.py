from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/detect_tables', methods=['POST'])
def detect_tables():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file
    filename = f"{uuid.uuid4().hex}.png"  # Ensure unique file name
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Detect tables in the image
    detected_tables, output_path = process_image(file_path)

    # Return the table detection results (bounding boxes) and the output image path
    return jsonify({
        'detected_tables': detected_tables,
        'result_image': output_path.split(os.sep)[-1]  # Return just the file name
    })

def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_tables = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w > 100 and h > 100:
            detected_tables.append({'bbox': (x, y, w, h)})
            
            # Draw the table bounding box in green
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green color for the table box

            last_column_x = x + w - 50  # Approximate position of the last column
            last_column_width = 50  # Width of the last column

            # Highlight the last column in red
            cv2.rectangle(image, (last_column_x, y), (last_column_x + last_column_width, y + h), (0, 0, 255), 3)  # Red color for the last column
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "Total of Amount", (last_column_x + 5, y + 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # To highlight the third row in the last column
            row_height = h // 5  # Divide the height into 5 rows (this can vary depending on the table structure)

            # Assuming that the third row is around 3/5 of the total height
            third_row_y = y + 2 * row_height

            # Draw the third row in the last column in blue
            cv2.rectangle(image, (last_column_x, third_row_y), (last_column_x + last_column_width, third_row_y + row_height), (255, 0, 0), 3)  # Blue color

    # Save the result image with highlighted tables
    output_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4().hex}.png")
    cv2.imwrite(output_path, image)

    return detected_tables, output_path

@app.route('/outputs/<filename>', methods=['GET'])
def serve_output_image(filename):
    # Serve the result image from the outputs folder
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
