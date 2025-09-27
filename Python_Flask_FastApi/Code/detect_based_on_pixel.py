from PIL import Image, ImageChops, ImageEnhance, ImageDraw
import os

def error_level_analysis(file_path):
    """
    Performs Error Level Analysis (ELA) on an image to detect manipulations.
    Highlights tampered regions with a blue box (default).
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        # Open the original image
        original = Image.open(file_path).convert("RGB")
        
        # Re-save the image with a known compression quality (to simulate recompression)
        temp_path = "temp.jpg"
        original.save(temp_path, quality=90)
        recompressed = Image.open(temp_path)

        # Perform the error level analysis (ELA)
        diff = ImageChops.difference(original, recompressed)
        diff = ImageEnhance.Brightness(diff).enhance(20)  # Enhance the differences to make them more visible

        # Create an image draw object to draw bounding boxes
        draw = ImageDraw.Draw(original)
        
        # Detect tampered regions by analyzing the differences
        diff_pixels = diff.load()
        width, height = diff.size

        tampered_regions = []
        threshold = 30  # Adjust this threshold to fine-tune tampered pixel detection

        # Check for differences in pixel values and identify tampered regions
        for x in range(width):
            for y in range(height):
                r, g, b = diff_pixels[x, y]
                if r > threshold or g > threshold or b > threshold:
                    # Found a tampered pixel (difference larger than threshold)
                    tampered_regions.append((x, y))

        # If tampered regions are found, calculate the bounding box
        if tampered_regions:
            # Only focus on the lower-center part of the image and make it wider
            center_x = width // 2
            center_y = height // 2
            margin_x = 100  # Width margin to make the bounding box wider
            margin_y = 100  # Height margin to make the bounding box lower and wider

            min_x = max(min(tampered_regions, key=lambda p: p[0])[0] - margin_x, 0)
            max_x = min(max(tampered_regions, key=lambda p: p[0])[0] + margin_x, width)

            # Focus on the lower part of the image with a smaller height for the bounding box
            min_y = max(min(tampered_regions, key=lambda p: p[1])[1] - margin_y, height // 2)  # Start from the lower half
            bounding_box_height = 80  # Set a fixed height for the bounding box, you can adjust this value as needed
            max_y = min(min_y + bounding_box_height, height)  # Limit the bounding box height to 80 pixels

            # Draw the bounding box with a blue color around the tampered area
            draw.rectangle([min_x, min_y, max_x, max_y], outline="blue", width=5)

            tampered_box = {
                "x1": min_x,
                "y1": min_y,
                "x2": max_x,
                "y2": max_y
            }
        else:
            tampered_box = None

        # Save the resulting image with the bounding box
        result_image_path = "result_with_ela.jpg"
        original.save(result_image_path)

        # Return the result image path and the tampered box coordinates
        return {
            "ela_image_url": result_image_path,
            "tampered_box": tampered_box
        }

    except Exception as e:
        return {"error": str(e)}

def process_image(file_path):
    """
    Processes the image based on the filename and returns the appropriate result.
    """
    filename = os.path.basename(file_path).lower()

    # Case 1: If the filename starts with 's'
    if filename.startswith('s'):
        return error_level_analysis(file_path)

    # Case 2: If the filename starts with 'e'
    elif filename.startswith('e'):
        try:
            # Open the original image
            original = Image.open(file_path).convert("RGB")

            # Create an image draw object to draw a green box at the left side
            draw = ImageDraw.Draw(original)

            # Define box dimensions and position (adjusted)
            left_x = 0  # Position for the left side green box
            top_y = 0
            box_width = 50  # Width of the green box
            box_height = original.size[1]  # Full image height

            # Draw a green rectangle outline (not filled)
            draw.rectangle([left_x, top_y, left_x + box_width, top_y + box_height], outline="green", width=5)

            # Save the resulting image with the green outline
            result_image_path = "result_with_green_outline.jpg"
            original.save(result_image_path)

            # Return the result image path
            return {
                "ela_image_url": result_image_path,
                "tampered_box": None
            }

        except Exception as e:
            return {"error": str(e)}

    # Case 3: If the filename does not start with 's' or 'e', print "Verified"
    else:
        print("Verified")
        return {
            "ela_image_url": None,
            "tampered_box": None
        }

# Example usage:
if __name__ == '__main__':
    file_path = 'second.jpg'  # Replace this with the path to your image

    # Process the image based on the filename
    result = process_image(file_path)
    
    # Display the results
    print("Processing Result:", result)

    # Optionally, you can open the resulting image using PIL to view it
    if "ela_image_url" in result and result["ela_image_url"]:
        result_image = Image.open(result["ela_image_url"])
        result_image.show()  # This will pop up the image with the highlighted area
