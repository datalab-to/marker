import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Transformers uses .isin for an op, which is not supported on MPS

from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor

def create_model_dict(device=None, dtype=None) -> dict:
    foundation_predictor = FoundationPredictor(device=device, dtype=dtype)
    return {
        "foundation_model": foundation_predictor,
        "layout_model": LayoutPredictor(device=device, dtype=dtype),
        "recognition_model": RecognitionPredictor(foundation_predictor),
        "table_rec_model": TableRecPredictor(device=device, dtype=dtype),
        "detection_model": DetectionPredictor(device=device, dtype=dtype),
        "ocr_error_model": OCRErrorPredictor(device=device, dtype=dtype)
    }