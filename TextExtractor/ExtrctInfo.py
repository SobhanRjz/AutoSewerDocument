import os
import cv2
import numpy as np
import torch
import easyocr
from PIL import Image
#import pytesseract
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as transforms
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

# Global model cache
_model_cache = {}
_cache_lock = threading.Lock()

class TextExtractor:
    """A class to extract text from video frames using multiple OCR engines"""

    def __init__(self, lazy_load=False, use_cache=True):
        """Initialize OCR models and configurations"""
        print("Initializing TextExtractor...")

        # Set device and optimize CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # Initialize Tesseract
        #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Get the directory where this file is located
        self.file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.OCR_model_path = os.path.join(self.file_dir, 'OCRModel')

        # Check if the local OCR model path exists
        if not os.path.exists(self.OCR_model_path):
            # If local path doesn't exist, use the remote model path
            self.OCR_model_path = 'ucaslcl/GOT-OCR2_0'
            print(f"Local OCR model not found. Using remote model: {self.OCR_model_path}")

        self.roi_coords = None

        # Model loading attributes
        self.easy_ocr = None
        self.tokenizer = None
        self.model = None
        self.transform = None

        # Try to load from cache first (if caching enabled)
        if use_cache:
            cache_key = self._get_cache_key()
            if self._load_from_cache(cache_key):
                print("Models loaded from cache")
                return

        if not lazy_load:
            self._load_models_sync()
        else:
            self._load_models_async()

        # Cache the loaded models (if caching enabled)
        if use_cache:
            self._save_to_cache(cache_key)

    def _get_cache_key(self):
        """Generate a cache key based on model paths and configuration"""
        # Create a unique key based on model path and device
        key_data = {
            'ocr_model_path': self.OCR_model_path,
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available()
        }
        key_str = str(key_data).encode('utf-8')
        return hashlib.md5(key_str).hexdigest()

    def _load_from_cache(self, cache_key):
        """Try to load models from cache"""
        with _cache_lock:
            if cache_key in _model_cache:
                cached_data = _model_cache[cache_key]
                try:
                    self.easy_ocr = cached_data['easy_ocr']
                    self.tokenizer = cached_data['tokenizer']
                    self.model = cached_data['model']
                    self.transform = cached_data['transform']
                    return True
                except (KeyError, AttributeError):
                    # Cache corrupted, remove it
                    del _model_cache[cache_key]
        return False

    def _save_to_cache(self, cache_key):
        """Save models to cache"""
        if self.easy_ocr and self.tokenizer and self.model and self.transform:
            with _cache_lock:
                try:
                    _model_cache[cache_key] = {
                        'easy_ocr': self.easy_ocr,
                        'tokenizer': self.tokenizer,
                        'model': self.model,
                        'transform': self.transform
                    }
                    print(f"Models cached with key: {cache_key[:8]}...")
                except Exception as e:
                    print(f"Failed to cache models: {e}")

    def _load_models_sync(self):
        """Load all models synchronously"""
        print("Loading OCR models...")

        start_time = time.time()

        # Initialize EasyOCR (fast)
        print("  Loading EasyOCR...")
        self.easy_ocr = easyocr.Reader(['en'], gpu=True)

        # Initialize GOT-OCR tokenizer first (needed for model)
        print("  Loading GOT-OCR tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.OCR_model_path,
            trust_remote_code=True,
            use_fast=True
        )

        print("  Loading GOT-OCR model...")
        self.model = self._initialize_got_ocr_model()

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        loading_time = time.time() - start_time
        print(f"All models loaded in {loading_time:.2f} seconds")

    def _load_models_async(self):
        """Load models asynchronously for better UX"""
        print("Starting asynchronous model loading...")

        # Load tokenizer first (needed for model initialization)
        print("  Loading GOT-OCR tokenizer...")
        self.tokenizer = self._load_tokenizer()

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit remaining loading tasks
            easyocr_future = executor.submit(self._load_easyocr)
            model_future = executor.submit(self._load_got_model)

            # Wait for completion with progress feedback
            print("  Loading EasyOCR...")
            self.easy_ocr = easyocr_future.result()

            print("  Loading GOT-OCR model...")
            self.model = model_future.result()

            # Image preprocessing pipeline (fast, do it here)
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        print("All models loaded asynchronously")

    def _load_easyocr(self):
        """Load EasyOCR model"""
        return easyocr.Reader(['en'], gpu=True)

    def _load_tokenizer(self):
        """Load GOT-OCR tokenizer"""
        return AutoTokenizer.from_pretrained(
            self.OCR_model_path,
            trust_remote_code=True,
            use_fast=True
        )

    def _load_got_model(self):
        """Load GOT-OCR model"""
        return self._initialize_got_ocr_model()

    def _ensure_models_loaded(self):
        """Ensure all models are loaded before use"""
        if self.easy_ocr is None or self.tokenizer is None or self.model is None:
            print("Models not loaded yet, loading now...")
            # Try cache first, then load if needed
            cache_key = self._get_cache_key()
            if not self._load_from_cache(cache_key):
                self._load_models_sync()
                self._save_to_cache(cache_key)

    @staticmethod
    def clear_model_cache():
        """Clear the global model cache"""
        with _cache_lock:
            _model_cache.clear()
            print("Model cache cleared")

    @staticmethod
    def get_cache_size():
        """Get the number of cached model instances"""
        with _cache_lock:
            return len(_model_cache)

    def _initialize_got_ocr_model(self):
        """Initialize and optimize the GOT-OCR model"""
        print(f"    Initializing GOT-OCR model from {self.OCR_model_path}...")

        proxies = {
            "http": "http://127.0.0.1:10808",
            "https": "http://127.0.0.1:10808",
        }

        # Use faster loading parameters
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }

        # Add device mapping for faster loading
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        model = AutoModel.from_pretrained(
            self.OCR_model_path,
            **model_kwargs
        )

        # Move to device faster
        if torch.cuda.is_available():
            model = model.to(self.device, non_blocking=True)
        else:
            model = model.to(self.device)

        #model.generation_config.cache_implementation = "static"
        model.half()
        model.eval()

        # Skip torch.compile() for faster loading (can be slow)
        if torch.__version__ >= "2.0":
            model = torch.compile(model)

        print("    GOT-OCR model initialized successfully")
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
        self._ensure_models_loaded()

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        raw_text = self.model.chat(self.tokenizer, image, ocr_type='ocr', gradio_input = True)
        return ''.join(c for c in raw_text if c.isdigit() or c == '.')

    def _extract_text_easyocr(self, image):
        """Extract text using EasyOCR"""
        self._ensure_models_loaded()

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

        # if save_crop:
        #     Image.fromarray(processed).save("cropped_image.png")

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