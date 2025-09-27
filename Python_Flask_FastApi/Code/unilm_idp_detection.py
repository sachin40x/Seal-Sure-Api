import requests
from transformers import pipeline
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Update this to your Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the Hugging Face pipeline for text classification (you can use other models or zero-shot classification)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to extract text using Tesseract OCR
def extract_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    custom_oem_psm_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(thresh, config=custom_oem_psm_config)
    return extracted_text, image

# Function to detect tampered words using Hugging Face
def detect_tampered_words(extracted_text, words, coords):
    labels = ["tampered", "original"]
    tampered_coords = []

    # Check if words is empty
    if not words:
        print("No words extracted from the image.")
        return tampered_coords

    # Use Hugging Face to classify each word in the extracted text
    for word, coord in zip(words, coords):
        if word.strip():  # Ensure the word is not empty
            print(f"Classifying word: {word}")  # Debug print
            try:
                result = classifier(word, candidate_labels=labels)
                label = result['labels'][0]  # Get the top predicted label
                if label == 'tampered':
                    tampered_coords.append(coord)
            except Exception as e:
                print(f"Error classifying word '{word}': {e}")

    return tampered_coords

# Function to find words and their bounding boxes
def find_words_coords(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    coords = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:  # Only consider words with non-zero confidence
            word = data['text'][i]
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            words.append(word)
            coords.append((x, y, w, h))
    return words, coords

# Function to apply pixelation effect on tampered words
def pixelate_region(image, x, y, w, h, pixel_size=10):
    # Crop the region of interest (ROI)
    roi = image[y:y+h, x:x+w]
    
    # Resize the ROI to a smaller size (pixelation)
    temp = cv2.resize(roi, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    
    # Resize it back to the original size
    pixelated_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Replace the original ROI with the pixelated one
    image[y:y+h, x:x+w] = pixelated_roi
    return image

# Function to highlight tampered words and pixelate them
def highlight_and_pixelate_tampered_words(image, coords):
    pixelated_image = image.copy()  # Copy to modify the pixelated version
    highlighted_image = image.copy()  # Copy to modify the highlighted version
    
    for coord in coords:
        x, y, w, h = coord
        # Apply pixelation effect to tampered words (on pixelated image)
        pixelated_image = pixelate_region(pixelated_image, x, y, w, h)
        
        # Optionally, add a red rectangle around tampered words for the highlighted image
        cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle
    
    return highlighted_image, pixelated_image

# Function to apply infrared effect on the image (specifically the pixelated one)
def apply_infrared_effect(image):
    # Convert to grayscale and increase red tones for infrared look
    infrared_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    infrared_image = cv2.applyColorMap(infrared_image, cv2.COLORMAP_JET)
    return infrared_image

# Main function
def main(image_path):
    # Step 1: Extract text from the image
    extracted_text, image = extract_text(image_path)
    print("Extracted Text:")
    print(extracted_text)

    # Step 2: Get words and their bounding boxes
    words, coords = find_words_coords(image)

    # Step 3: Detect tampered words using Hugging Face model
    tampered_coords = detect_tampered_words(extracted_text, words, coords)
    
    if tampered_coords:
        print(f"Tampered words detected at coordinates: {tampered_coords}")
    else:
        print("No tampered words detected.")

    # Step 4: Highlight and pixelate the tampered words in the image
    highlighted_image, pixelated_image = highlight_and_pixelate_tampered_words(image, tampered_coords)

    # Step 5: Apply infrared effect to the pixelated image
    infrared_pixelated_image = apply_infrared_effect(pixelated_image)

    # Step 6: Display the images in separate windows
    plt.figure(figsize=(12, 6))
    
    # Show the highlighted image
    plt.subplot(1, 2, 1)
    highlighted_image_rgb = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    plt.imshow(highlighted_image_rgb)
    plt.title("Highlighted Tampered Words")
    plt.axis('off')
    
    # Show the infrared pixelated image
    plt.subplot(1, 2, 2)
    plt.imshow(infrared_pixelated_image)
    plt.title("Pixelated Tampered Words (Infrared Effect)")
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    image_path = "live_images/done.jpg"  # Change to your image path
    main(image_path)
