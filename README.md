# Vehicle-Number-Plate-Detection
In order to identify and detect number plates in real-time video feeds, this project makes use of optical character recognition (OCR) and computer vision algorithms. 

It preprocesses each frame using the OpenCV and Pytesseract libraries, converting it to grayscale, applying adaptive thresholding, and carrying out morphological procedures to separate possible number plate regions. 

To find potential number plates, contours are then removed and filtered according to size and aspect ratio. After choosing the largest candidate plate, alphanumeric characters are extracted from the plate region using Tesseract OCR. 

Ultimately, the identified license plate and its associated text are appended to the initial frame, offering the viewer a visual representation of the information. 

This study shows how computer vision and optical character recognition (OCR) techniques can be used in real-world surveillance to automatically recognize license plates.
