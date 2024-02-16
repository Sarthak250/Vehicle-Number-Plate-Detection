import cv2
import pytesseract
import os

# Set the path to your Tesseract OCR installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Change this path

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph

def is_license_plate(contour):
    # Calculate aspect ratio of the bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h

    # Define aspect ratio criteria for a license plate
    min_aspect_ratio = 2
    max_aspect_ratio = 6

    # Filter contours based on aspect ratio
    return min_aspect_ratio <= aspect_ratio <= max_aspect_ratio

def is_number_plate_candidate(contour):
    # Filter contours based on area and aspect ratio
    min_area = 1000  # Increase this value based on your specific case
    return cv2.contourArea(contour) > min_area and is_license_plate(contour)

def detect_and_recognize_number_plate(video_source=0, save_path="detected_frames"):
    # Create the save path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Open a connection to the front camera (0 represents the default camera, you can change it if you have multiple cameras)
    cap = cv2.VideoCapture(video_source)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture a single frame
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Preprocess the frame
        processed_frame = preprocess_image(frame)

        # Find contours in the processed frame
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find potential number plate regions
        potential_plates = [cnt for cnt in contours if is_number_plate_candidate(cnt)]

        # Sort potential plates by size in descending order
        potential_plates = sorted(potential_plates, key=cv2.contourArea, reverse=True)

        # Use the largest potential plate as the number plate
        if potential_plates:
            # Get the bounding box for the largest potential plate
            x, y, w, h = cv2.boundingRect(potential_plates[0])

            # Extract the region of interest (ROI) containing the potential plate
            plate_roi = frame[max(0, y-20):y + h+20, max(0, x-20):x + w+20]

            # Use Tesseract OCR to read text from the plate ROI
            custom_config = r'--oem 3 --psm 6 outputbase alphanumeric'  # Adjust OCR configuration as needed
            number_plate_text = pytesseract.image_to_string(plate_roi, config=custom_config)

            # Display the frame with the detected number plate and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, number_plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the frame with annotated number plate
            save_filename = os.path.join(save_path, f"detected_frame.png")
            cv2.imwrite(save_filename, frame)

            # Display the frame with the number plate only
            cv2.imshow("Number Plate Detection", plate_roi)

        # Display the original frame
        cv2.imshow("Original Frame", frame)

        # Check for the 'q' key to quit the program
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_recognize_number_plate()
