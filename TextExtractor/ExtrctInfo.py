import os
import cv2
import numpy as np
import torch
import easyocr
from PIL import Image
import pytesseract
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as transforms

class TextExtractor:
    """A class to extract text from video frames using multiple OCR engines"""

    def __init__(self):
        """Initialize OCR models and configurations"""
        # Set device and optimize CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # Initialize Tesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Initialize EasyOCR
        self.easy_ocr = easyocr.Reader(['en'], gpu=True)

        # Initialize GOT-OCR model
        proxies = {
            "http": "http://127.0.0.1:10808",
            "https": "http://127.0.0.1:10808",
        }
    
        # Get the directory where this file is located
        self.file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.OCR_model_path = os.path.join(self.file_dir, 'OCRModel')

        # Check if the local OCR model path exists
        if not os.path.exists(self.OCR_model_path):
            # If local path doesn't exist, use the remote model path
            self.OCR_model_path = 'ucaslcl/GOT-OCR2_0'
            print(f"Local OCR model not found. Using remote model: {self.OCR_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.OCR_model_path, 
            trust_remote_code=True,
            use_fast = True
        )

        self.model = self._initialize_got_ocr_model()

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.roi_coords = None

    def _initialize_got_ocr_model(self):
        """Initialize and optimize the GOT-OCR model"""
        proxies = {
            "http": "http://127.0.0.1:10808",
            "https": "http://127.0.0.1:10808",
        }
        model = AutoModel.from_pretrained(
            self.OCR_model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.device,
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device, non_blocking=True)

        model.half()
        model.eval()
        
        if torch.__version__ >= "2.0":
            model = torch.compile(model)
            
        return model

    def _read_video_frame(self, video_path):
        """Read the first frame from a video file"""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        return ret, frame

    def _preprocess_image(self, image):
        """Enhance image quality for better OCR results"""
        # Convert to numpy array
        img_array = np.array(image)

        # Denoise
        denoised = cv2.GaussianBlur(img_array, (5, 5), 0)

        # Sharpen
        kernel = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        return sharpened

    def _get_user_roi(self, image, full_image=False):
        """Select region of interest from image"""
        if full_image:
            height, width = image.shape[:2]
            self.roi_coords = (0, 0, width, height)
            return self.roi_coords

        # Create window with instructions
        window_name = "Select Region of Interest"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 768)
        
        # Set window to be always on top
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        # Add instructions overlay
        img_with_text = image.copy()
        instructions = [
            "Instructions:",
            "1. Click and drag to select region", 
            "2. Press SPACE or ENTER to confirm",
            "3. Press C to cancel selection",
            "4. Press ESC to exit"
        ]

        # Add semi-transparent overlay
        overlay = img_with_text.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, img_with_text, 0.7, 0, img_with_text)

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 40
        for line in instructions:
            cv2.putText(img_with_text, line, (20, y), font, 0.7, (255, 255, 255), 2)
            y += 25

        # Show window and bring to front
        cv2.imshow(window_name, img_with_text)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        # Get ROI with enhanced selectROI
        self.roi_coords = cv2.selectROI(window_name, img_with_text, False)
        cv2.destroyAllWindows()
        return self.roi_coords

    @torch.inference_mode()
    def _extract_text_Complete(self, image):
        """Extract text using GOT-OCR model"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        raw_text = self.model.chat(self.tokenizer, image, ocr_type='ocr')
        return ''.join(c for c in raw_text if c.isdigit() or c == '.')

    def _extract_text_easyocr(self, image):
        """Extract text using EasyOCR"""
        image_array = np.array(image)
        result = self.easy_ocr.readtext(image_array, detail=0)

        if not result:
            return ""

        text = result[0].replace(" ", "").replace(",", ".")
        return ''.join(c for c in text if c.isdigit() or c == '.')

    def extract_text_from_video_frame(self, frame=None, frame_path=None, save_crop=True, UseFullOCR=True):
        """Extract text from video frame using selected ROI"""
        if frame_path:
            frame = cv2.imread(frame_path)

        if frame is None:
            raise ValueError("No frame provided")

        if self.roi_coords is None:
            raise ValueError("No ROI coordinates found. Please select ROI first.")

        # Convert and crop frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        x0, y0, w, h = self.roi_coords
        cropped = image.crop((x0, y0, x0 + w, y0 + h))
        processed = self._preprocess_image(cropped)

        if save_crop:
            Image.fromarray(processed).save("cropped_image.png")

        return (self._extract_text_Complete if UseFullOCR else self._extract_text_easyocr)(processed)



if __name__ == "__main__":
    # Example usage
    VIDEO_PATH = r"C:\Users\sobha\Desktop\detectron2\Data\TestFilm\Closed circuit television (CCTV) sewer inspection.mp4"
    
    extractor = TextExtractor()
    success, frame = extractor._read_video_frame(VIDEO_PATH)
    
    if success:
        extractor._get_user_roi(frame, full_image=False)
        text = extractor.extract_text_from_video_frame(frame, save_crop=False)
        print(f"Extracted text: {text}")
    else:
        print("Failed to read video frame")