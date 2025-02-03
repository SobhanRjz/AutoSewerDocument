import cv2
from PIL import Image
import pytesseract
from PIL import ImageEnhance, ImageFilter
import numpy as np
from paddleocr import PaddleOCR

class TextExtractor:
    """A class to extract text from video frames using multiple OCR engines"""
    
    def __init__(self):
        """Initialize the TextExtractor with OCR configurations"""
        # Set Tesseract executable path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # Initialize PaddleOCR
        self.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True)
        
    def _read_video_frame(self, video_path):
        """Read the first frame from a video file"""
        video_capture = cv2.VideoCapture(video_path)
        ret, frame = video_capture.read()
        video_capture.release()
        return ret, frame
    
    def _preprocess_image(self, image):
        """Enhance image quality for better text extraction."""
        
        # Convert PIL image to grayscale (if not already)
        gray = image.convert('L')

        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(gray)
        contrast_img = contrast_enhancer.enhance(2.5)  # Increased contrast
        
        # Enhance sharpness
        sharp_enhancer = ImageEnhance.Sharpness(contrast_img)
        sharp_img = sharp_enhancer.enhance(2.5)  # Increased sharpness
        
        # Convert to OpenCV format
        img_cv = np.array(sharp_img)

        # Apply Gaussian Blur to reduce noise
        img_blur = cv2.GaussianBlur(img_cv, (3,3), 0)

        # Apply adaptive thresholding instead of fixed threshold
        threshold_img = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

        # Apply morphological operations to clean up noise
        kernel = np.ones((1,1), np.uint8)
        morph_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)
        
        # Histogram equalization to improve contrast
        hist_eq_img = cv2.equalizeHist(morph_img)

        # Display the processed image
        # cv2.imshow('Processed Image', hist_eq_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return Image.fromarray(hist_eq_img)  # Convert back to PIL for OCR
    
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
            return (0, 0, width, height)
            
        cv2.namedWindow("Select Region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Region", 800, 600)
        roi = cv2.selectROI("Select Region", display_img)
        cv2.destroyAllWindows()
        return roi
    
    def _extract_text(self, image):
        """Extract text using multiple OCR engines"""
        # Tesseract OCR
        #tesseract_text = pytesseract.image_to_string(image)
        
        # PaddleOCR
        paddle_image = np.array(image)
        paddle_result = self.paddle_ocr.ocr(paddle_image, cls=True)
        paddle_text = '\n'.join([line[1][0] for line in paddle_result[0]]) if paddle_result[0] else ''
        
        return paddle_text
    
    def extract_text_from_video_frame(self, frame_path, save_crop=False):
        """
        Extract text from a specific region of video frames
        

        Args:
            frames: List of video frames
            save_crop (bool): Whether to save the cropped image
        
        Returns:
            str: Combined extracted text from multiple OCR engines
        """
        all_texts = []
        frame = cv2.imread(frame_path)

        # Convert frame to RGB and PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        display_frame = frame.copy()
        
        # Get ROI from user
        roi = self._get_user_roi(display_frame, full_image=True)
        x0, y0, w, h = roi
        x1, y1 = x0 + w, y0 + h
        
        # Crop and process image
        cropped_image = image.crop((x0, y0, x1, y1))
        processed_image = self._preprocess_image(cropped_image)
        
        # Extract text
        paddle_text = self._extract_text(processed_image)
        
        # Save processed crop if requested
        if save_crop:
            processed_image.save("cropped_image.png")
        
        return paddle_text



if __name__ == "__main__":
    # Example usage
    video_path = r"C:\Users\sobha\Desktop\detectron2\Data\TestFilm\Closed circuit television (CCTV) sewer inspection.mp4"
    
    extractor = TextExtractor()
    extracted_text = extractor.extract_text_from_video_frame(video_path, save_crop=False)
    print("Extracted text:", extracted_text)