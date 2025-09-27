import cv2
import numpy as np
from pyzbar.pyzbar import decode
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Enable CORS for frontend at localhost:5173

def detect_and_verify_barcode(image_bytes, document_data):
    # Convert image bytes to a numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Use pyzbar to decode barcode data
    barcodes = decode(img)  # Decode barcodes (including QR codes)
    
    if barcodes:
        barcode_data = [barcode.data.decode('utf-8') for barcode in barcodes]  # Get barcode data as a list of strings
        return {'barcode_data': barcode_data, 'status': 'success'}
    else:
        return {'error': 'No barcode detected', 'status': 'failure'}

@app.route('/verify-code', methods=['POST'])  # Ensure the route matches the frontend request
def verify_barcode():
    try:
        # Retrieve the image file from the request
        image = request.files['image']
        image_bytes = image.read()  # Read image as bytes
        
        # You can add any additional document data here if needed
        document_data = {}

        # Call the barcode detection function
        result = detect_and_verify_barcode(image_bytes, document_data)

        # Return the result as JSON
        return jsonify(result)

    except Exception as e:
        # Return error message in case of exception
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)  # Ensure it runs on all network interfaces and port 5000
