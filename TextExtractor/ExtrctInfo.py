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
        self.tokenizer = AutoTokenizer.from_pretrained(
            'ucaslcl/GOT-OCR2_0', 
            trust_remote_code=True,
            use_fast = True,
            proxies=proxies
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
            'ucaslcl/GOT-OCR2_0',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.device,
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            proxies=proxies
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

        cv2.namedWindow("Select Region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Region", 800, 600)
        self.roi_coords = cv2.selectROI("Select Region", image)
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

    def extract_text_from_video_frame(self, frame=None, frame_path=None, save_crop=True, UseGOTOCR=True):
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

        return (self._extract_text_Complete if UseGOTOCR else self._extract_text_easyocr)(processed)



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