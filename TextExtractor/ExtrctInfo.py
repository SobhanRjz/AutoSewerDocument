import cv2
from PIL import Image
import pytesseract
from PIL import ImageEnhance, ImageFilter
import numpy as np
from paddleocr import PaddleOCR
import sys
# sys.path.append(r'C:\Users\sobha\Desktop\detectron2\Code\EasyOCR\easyocr')
# import easyocr
import keras_ocr
#from calamari_ocr.ocr.predict.predictor import MultiPredictor, PredictorParams

import os

#from doctr.models import ocr_predictor
# import doctr.io
# from doctr.io import DocumentFile
class TextExtractor:



    """A class to extract text from video frames using multiple OCR engines"""
    



    def __init__(self):
        """Initialize the TextExtractor with OCR configurations"""
        # Set Tesseract executable path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # Initialize Calamari OCR with models
        model_dir = r"C:\Users\sobha\Desktop\detectron2\Code\Calamari-OCR calamari_models master uw3-modern-english"
        # Initialize DocTR model
        #self.doctr_model =ocr_predictor(det_arch='db_resnet50', reco_arch='vitstr_base', pretrained=True)
        self.paddle_ocr = PaddleOCR(lang='en',use_angle_cls=True, lang_list=['en'],use_gpu=True)

        self.roi_coords = None
        
    def _read_video_frame(self, video_path):
        """Read the first frame from a video file"""
        video_capture = cv2.VideoCapture(video_path)
        ret, frame = video_capture.read()
        video_capture.release()
        return ret, frame
    
    def _preprocess_image(self, cropped_image):
        """
        Preprocess an image to improve OCR accuracy:
        - Converts to grayscale
        - Applies Otsu's thresholding 
        - Uses distance transform
        - Applies morphological operations
        - Extracts text using contours & convex hull
        - Normalizes for deep learning OCR models
        """
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding to get white text on black background
        thresh = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # Removed _INV to get white text

        # Apply distance transform with reduced parameters for less blurring
        dist = cv2.distanceTransform(255 - thresh, cv2.DIST_L2, 3)  # Invert thresh for distance transform
        # Normalize with adjusted range to preserve more detail
        dist = cv2.normalize(dist, None, 0, 0.8, cv2.NORM_MINMAX)  # Reduced upper range to 0.8
        # Scale to 8-bit with gamma correction to enhance contrast
        gamma = 0.7
        dist = np.power(dist, gamma) * 255
        dist = np.clip(dist, 0, 255).astype("uint8")

        # Threshold distance transform using Otsu's method
        dist = cv2.threshold(dist, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Morphological opening to clean text
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)

        # Find contours to detect text areas
        cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        chars = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if w >= 35 and h >= 100:  # Keep only text-like contours
                chars.append(c)

        # Create convex hull around detected text
        if chars:
            chars = np.vstack([chars[i] for i in range(0, len(chars))])
            hull = cv2.convexHull(chars)

            # Create a mask and draw the convex hull
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [hull], -1, 255, -1)

            # Dilate to further highlight text areas
            mask = cv2.dilate(mask, None, iterations=2)

            # Bitwise AND to extract just text regions
            final = cv2.bitwise_and(thresh, thresh, mask=mask)  # Use thresh instead of opening
        else:
            final = thresh  # If no characters found, use thresholded image

        cv2.imwrite("9_final.png", final)


        return final
    def _get_user_roi(self, display_img, full_image=False):
        """Let user select region of interest or use full image
        
        Args:
            display_img: Image to display for ROI selection
            full_image: If True, returns ROI for full image. If False, lets user select ROI.
        
        Returns:
            tuple: ROI coordinates (x, y, width, height)
        """
        if full_image:
            height, width = display_img.shape[:2]
            roi = (0, 0, width, height)
            self.roi_coords = roi
            return roi
            
        cv2.namedWindow("Select Region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Region", 800, 600)
        roi = cv2.selectROI("Select Region", display_img)
        cv2.destroyAllWindows()
        self.roi_coords = roi
        return roi
    def _extract_text_keras(self, image):
        """Extract text using Keras-OCR engine
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Extracted text from Keras-OCR
        """
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Make prediction
        predictions = self.keras_pipeline.recognize([img_array])
        
        # Extract text from predictions
        extracted_text = ''
        if predictions:
            for pred_group in predictions[0]:
                text = pred_group[0]  # Get text from prediction tuple
                extracted_text += text + ' '
                
        return extracted_text.strip()
    def _extract_text_calamari(self, image):
        """Extract text using Calamari OCR engine
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Extracted text from Calamari OCR
        """
        # Save image temporarily since Calamari requires file path
        temp_path = "temp_image.png"
        image.save(temp_path)

        # Set up predictor

        # Make prediction
        #predictions = self.predictor.predict_file(temp_path)
        
        # Read image for raw prediction
        raw_image = cv2.imread(temp_path)
        raw_image_generator = [raw_image]
        
        # Process raw image
        for sample in self.predictor.predict_raw(raw_image_generator):
            inputs, prediction, meta = sample.inputs, sample.outputs, sample.meta


        # Extract text from predictions
        # extracted_text = ""
        # for prediction in predictions:
        #     extracted_text += prediction.sentence + " "

        # Clean up temp file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return extracted_text.strip()
    def _extract_text_doctr(self, image):
       # Check if image is None
        if image is None:
            raise ValueError("Error: The image is None. Ensure the video frame is captured properly.")

        # Convert PIL image to NumPy array if needed
        if isinstance(image, np.ndarray) is False:
            image = np.array(image)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        result = self.doctr_model([image])
        
        # Print detected text
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    print("Detected text:", " ".join([word.value for word in line.words]))
                
        return extracted_text.strip()
    def _extract_text(self, image):
        """Extract text using multiple OCR engines"""
        # Tesseract OCR
        #tesseract_text = pytesseract.image_to_string(image)
        
        # PaddleOCR
        paddle_image = np.array(image)
        paddle_result = self.paddle_ocr.ocr(paddle_image, cls=True)
        # Post-processing: Keep only numbers and 'm'
        allowed_chars = "0123456789.mM"
        filtered_text = ""

        if paddle_result is not None:
            for line in paddle_result:
                if line is not None:
                    for word_info in line:
                        if word_info is not None and len(word_info) > 1 and word_info[1] is not None:
                            word = word_info[1][0]  # Extract detected word
                            filtered_text += "".join([char for char in word if char in allowed_chars]) + " "
        else:
            filtered_text = ""
            print("No text detected")
        # Extract only numbers from filtered text
        
        numbers = ''.join(char for char in filtered_text if char.isdigit() or char == '.')
        return numbers.strip()
    
    def _extract_text_easyocr(self, imagepath):
        """Extract text using EasyOCR engine
        

        Args:
            imagepath: Path to the image file
            
        Returns:
            str: Extracted text from EasyOCR
        """
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en']) 
        
        # Read text from image
        result = reader.readtext(imagepath)
        
        # Extract text from result
        extracted_text = ''     
        for detection in result:
            text = detection[1]  # Get the text content
            extracted_text += text + ' '
        
        return extracted_text.strip()
    def extract_text_from_video_frame(self, frame = None, frame_path = None, save_crop=False):
        """
        Extract text from a specific region of video frames using stored ROI coordinates



        Args:
            frame_path: Path to the video frame image
            save_crop (bool): Whether to save the cropped image
        
        Returns:
            str: Extracted text from OCR engine
        """
        if frame_path is None:
            frame = frame
        else:
            frame = cv2.imread(frame_path)


        # Convert frame to RGB and PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Use stored ROI coordinates
        if self.roi_coords is not None:
            x0, y0, w, h = self.roi_coords
            x1, y1 = x0 + w, y0 + h
            
            # Crop and process image
            cropped_image = image.crop((x0, y0, x1, y1))
            processed_image = self._preprocess_image(cropped_image)
            
            # Extract text
            paddle_text = self._extract_text(processed_image)
            #easyocr_text = self._extract_text_easyocr(frame_path)
            #keras_text = self._extract_text_keras(processed_image)
            #calamari_text = self._extract_text_calamari(processed_image)
            #doctr_text = self._extract_text_doctr(processed_image)

            # Save processed crop if requested
            if save_crop:
                processed_image.save("cropped_image.png")
            
            return paddle_text
        else:
            self.logger.warning("No ROI coordinates found. Please select ROI first.")
            return ""



if __name__ == "__main__":
    # Example usage
    video_path = r"C:\Users\sobha\Desktop\detectron2\Data\TestFilm\Closed circuit television (CCTV) sewer inspection.mp4"
    
    extractor = TextExtractor()
    # Get first frame from video
    ret, frame = extractor._read_video_frame(video_path)
    if ret:
        # Pass frame to get user ROI
        extractor._get_user_roi(frame, full_image=False)
        extracted_text = extractor.extract_text_from_video_frame(frame, save_crop=False)
    else:
        print("Failed to read video frame")

    
    print("Extracted text:", extracted_text)