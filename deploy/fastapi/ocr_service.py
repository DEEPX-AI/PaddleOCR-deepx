#!/usr/bin/env python3
"""
Simple OCR Service using PaddleOCR with FastAPI
Provides REST API for OCR inference with Baidu AI Studio compatibility
"""

import os
import base64
import json
import uuid
import sys
import math
import random
import PIL
from io import BytesIO
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from paddleocr import PaddleOCR
from paddleocr import DocImgOrientationClassification, TextImageUnwarping
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import uvicorn
import cv2

# ============================================================================
# Model Path Configuration
# ============================================================================

def get_model_paths():
    """
    Get model paths based on USE_MOBILE environment variable
    Returns dict with detection and recognition model paths
    """
    use_mobile = os.getenv('USE_MOBILE', 'false').lower() == 'true'
    models_dir = Path.home() / '.paddlex' / 'official_models'
    
    if use_mobile:
        det_model_name = 'PP-OCRv5_mobile_det'
        rec_model_name = 'PP-OCRv5_mobile_rec'
    else:
        det_model_name = 'PP-OCRv5_server_det'
        rec_model_name = 'PP-OCRv5_server_rec'
    
    det_model_path = models_dir / det_model_name
    rec_model_path = models_dir / rec_model_name
    
    # Check if models exist
    result = {
        'det_model_name': det_model_name if det_model_path.exists() else None,
        'det_model_dir': str(det_model_path) if det_model_path.exists() else None,
        'rec_model_name': rec_model_name if rec_model_path.exists() else None,
        'rec_model_dir': str(rec_model_path) if rec_model_path.exists() else None,
        'model_type': 'mobile' if use_mobile else 'server'
    }
    
    return result

# ============================================================================
# DEEPX NPU Support - Single Model Set with Dual Instances
# ============================================================================

# Global NPU models (loaded once and shared)
_npu_models = None

# Global NPU OCR instances (one sync, one async)
_npu_ocr_sync_instance = None
_npu_ocr_async_instance = None

def load_npu_models_once():
    """
    Load NPU models once and store globally
    All models are loaded when first --deepx request arrives
    
    Returns:
        dict: Dictionary containing all loaded models
    """
    global _npu_models
    
    if _npu_models is not None:
        return _npu_models
    
    print("🔧 Loading DEEPX NPU models (ONE-TIME INITIALIZATION)...")
    
    # Import dx_engine
    from dx_engine import InferenceEngine as IE
    
    # Get model type from environment
    use_mobile = os.getenv('USE_MOBILE', 'false').lower() == 'true'
    model_type = 'mobile' if use_mobile else 'server'
    
    # Import deepx engine
    # Use DEEPX_PATH environment variable if set, otherwise use relative path (for local development)
    deepx_path_str = os.getenv('DEEPX_PATH')
    if deepx_path_str:
        deepx_path = Path(deepx_path_str)
    else:
        deepx_path = Path(__file__).parent / 'deepx'
    
    if not deepx_path.exists():
        error_msg = f"DEEPX NPU not available: deepx directory not found at {deepx_path}. Please set DEEPX_PATH environment variable or ensure deepx folder is present."
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=503, detail=error_msg)
    
    sys.path.insert(0, str(deepx_path))
    
    # Model directory
    model_dir = deepx_path / 'engine' / 'model_files' / model_type
    
    if not model_dir.exists():
        error_msg = f"DEEPX NPU not available: NPU model directory not found at {model_dir}. Please download NPU models or build Docker image with --deepx flag."
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=503, detail=error_msg)
    
    print(f"   Model type: {model_type}")
    print(f"   Model directory: {model_dir}")
    
    # Load models
    def load_model(name):
        path = model_dir / name
        if not path.exists():
            error_msg = f"DEEPX NPU not available: NPU model file not found: {path}. Please download NPU models using local_deepx_setup.sh or build Docker with --deepx flag."
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=503, detail=error_msg)
        print(f"   Loading: {name}")
        try:
            return IE(str(path))
        except Exception as e:
            error_msg = f"DEEPX NPU model loading failed for {name}: {e}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=503, detail=error_msg)
    
    # Detection models (640, 960)
    det_prefix = 'det_mobile' if model_type == 'mobile' else 'det_v5'
    det_models = {
        640: load_model(f"{det_prefix}_640.dxnn"),
        960: load_model(f"{det_prefix}_960.dxnn")
    }
    
    # Classification model
    cls_model = load_model("textline_ori.dxnn")
    
    # Recognition models (ratio 3, 5, 10, 15, 25, 35)
    rec_prefix = 'rec_mobile' if model_type == 'mobile' else 'rec_v5'
    rec_models = {
        ratio: load_model(f"{rec_prefix}_ratio_{ratio}.dxnn")
        for ratio in [3, 5, 10, 15, 25, 35]
    }
    
    # Document preprocessing models
    doc_ori_model = load_model("doc_ori_fixed.dxnn")
    doc_unwarping_model = load_model("UVDoc_pruned_p3.dxnn")
    
    # Dictionary
    dict_path = model_dir / 'ppocrv5_dict.txt'
    if not dict_path.exists():
        error_msg = f"DEEPX NPU not available: Dictionary file not found: {dict_path}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=503, detail=error_msg)
    
    _npu_models = {
        'det_models': det_models,
        'cls_model': cls_model,
        'rec_models': rec_models,
        'doc_ori_model': doc_ori_model,
        'doc_unwarping_model': doc_unwarping_model,
        'dict_path': str(dict_path),
        'model_type': model_type
    }
    
    print(f"✅ All NPU models loaded successfully ({model_type})")
    print(f"   Models will be reused for all requests")
    
    return _npu_models

def get_npu_ocr_instance(
    sync: bool = False,
    use_doc_orientation: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = True,
    det_db_thresh: float = 0.3,
    det_db_box_thresh: float = 0.6,
    det_db_unclip_ratio: float = 1.5,
    rec_score_thresh: float = 0.0
):
    """
    Get or create DEEPX NPU OCR instance (NO CACHING)
    
    Models are loaded once globally at first use
    Separate sync and async instances are created once
    Both instances share the same pre-loaded models
    
    Args:
        sync: Use sync PaddleOcr (True) or async AsyncPipelineOCR (False)
        use_doc_orientation: Enable document orientation classification (NPU)
        use_doc_unwarping: Enable document unwarping (NPU)
        use_textline_orientation: Enable textline orientation classification (NPU)
        det_db_thresh: Detection threshold
        det_db_box_thresh: Detection box threshold
        det_db_unclip_ratio: Detection unclip ratio
        rec_score_thresh: Recognition score threshold
    """
    global _npu_models, _npu_ocr_sync_instance, _npu_ocr_async_instance
    
    # Load models once (if not already loaded)
    models = load_npu_models_once()
    
    # Get mode name
    mode_name = "sync" if sync else "async"
    
    # Debug
    print(f"🔍 NPU OCR Request:")
    print(f"   Mode: {mode_name}")
    print(f"   Model type: {models['model_type']}")
    print(f"   Config: doc_ori={use_doc_orientation}, unwarping={use_doc_unwarping}, textline_ori={use_textline_orientation}")
    
    # Get existing instance or create new one
    if sync:
        if _npu_ocr_sync_instance is None:
            print(f"🔄 Creating sync NPU instance (using pre-loaded models)...")
            _npu_ocr_sync_instance = _create_npu_instance(
                sync=True,
                models=models,
                use_doc_orientation=use_doc_orientation,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                det_db_thresh=det_db_thresh,
                det_db_box_thresh=det_db_box_thresh,
                det_db_unclip_ratio=det_db_unclip_ratio,
                rec_score_thresh=rec_score_thresh
            )
        else:
            print(f"✅ Using existing sync NPU instance")
        
        npu_ocr = _npu_ocr_sync_instance
    else:
        if _npu_ocr_async_instance is None:
            print(f"🔄 Creating async NPU instance (using pre-loaded models)...")
            _npu_ocr_async_instance = _create_npu_instance(
                sync=False,
                models=models,
                use_doc_orientation=use_doc_orientation,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                det_db_thresh=det_db_thresh,
                det_db_box_thresh=det_db_box_thresh,
                det_db_unclip_ratio=det_db_unclip_ratio,
                rec_score_thresh=rec_score_thresh
            )
        else:
            print(f"✅ Using existing async NPU instance")
        
        npu_ocr = _npu_ocr_async_instance
    
    # Update runtime flags (CRITICAL: respect endpoint parameters)
    # use_doc_preprocessing should be True if EITHER orientation OR unwarping is requested
    npu_ocr.use_doc_preprocessing = use_doc_orientation or use_doc_unwarping
    npu_ocr.use_textline_orientation = use_textline_orientation
    
    # Update doc_preprocessing object flags (unified for both sync and async)
    # Both PaddleOcr and AsyncPipelineOCR now use doc_preprocessing object
    if npu_ocr.doc_preprocessing is not None:
        npu_ocr.doc_preprocessing.use_doc_orientation = use_doc_orientation
        npu_ocr.doc_preprocessing.use_doc_unwarping = use_doc_unwarping
    
    print(f"   Runtime flags updated: preprocessing={npu_ocr.use_doc_preprocessing}, orientation={use_doc_orientation}, unwarping={use_doc_unwarping}")
    
    return npu_ocr

def _create_npu_instance(
    sync: bool,
    models: dict,
    use_doc_orientation: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
    det_db_thresh: float,
    det_db_box_thresh: float,
    det_db_unclip_ratio: float,
    rec_score_thresh: float
):
    """
    Create NPU OCR instance from pre-loaded models
    
    Args:
        sync: Create sync PaddleOcr (True) or async AsyncPipelineOCR (False)
        models: Pre-loaded model dictionary
        ... other OCR parameters
    
    Returns:
        PaddleOcr or AsyncPipelineOCR instance
    """
    try:
        # Import deepx engine
        # Use DEEPX_PATH environment variable if set, otherwise use relative path (for local development)
        deepx_path_str = os.getenv('DEEPX_PATH')
        if deepx_path_str:
            deepx_path = Path(deepx_path_str)
        else:
            # New location: deploy/fastapi/deepx
            deepx_path = Path(__file__).parent / 'deepx'
        
        if not deepx_path.exists():
            raise ImportError(f"deepx not found at {deepx_path}")
        
        sys.path.insert(0, str(deepx_path))
        
        if sync:
            from engine.paddleocr import PaddleOcr
            print("   Initializing DEEPX PaddleOcr (sync)...")
        else:
            from engine.paddleocr import AsyncPipelineOCR
            print("   Initializing DEEPX AsyncPipelineOCR (async)...")
        
        # Create instance using pre-loaded models
        if sync:
            # Sync mode: PaddleOcr
            npu_ocr = PaddleOcr(
                det_model=models['det_models'],
                cls_model=models['cls_model'],
                rec_models=models['rec_models'],
                rec_dict_dir=models['dict_path'],
                doc_ori_model=models['doc_ori_model'],
                doc_unwarping_model=models['doc_unwarping_model'],
                use_doc_preprocessing=True,  # Always enable - controlled at runtime
                use_doc_orientation=True,    # Always enable - controlled at runtime
                use_textline_orientation=use_textline_orientation,
                det_db_thresh=det_db_thresh,
                det_db_box_thresh=det_db_box_thresh,
                det_db_unclip_ratio=det_db_unclip_ratio,
                rec_score_thresh=rec_score_thresh
            )
            print(f"   ✅ DEEPX PaddleOcr (sync) created with pre-loaded models")
        else:
            # Async mode: AsyncPipelineOCR
            npu_ocr = AsyncPipelineOCR(
                det_model=models['det_models'],
                cls_model=models['cls_model'],
                rec_models=models['rec_models'],
                rec_dict_dir=models['dict_path'],
                doc_ori_model=models['doc_ori_model'],
                doc_unwarping_model=models['doc_unwarping_model'],
                use_doc_preprocessing=True,  # Always enable - controlled at runtime
                use_doc_orientation=True,    # Always enable - controlled at runtime
                use_textline_orientation=use_textline_orientation,
                det_db_thresh=det_db_thresh,
                det_db_box_thresh=det_db_box_thresh,
                det_db_unclip_ratio=det_db_unclip_ratio,
                rec_score_thresh=rec_score_thresh,
                input_interval=0.0,
                verbose=False
            )
            print(f"   ✅ DEEPX AsyncPipelineOCR (async) created with pre-loaded models")
        
        print(f"   Detection thresholds: thresh={det_db_thresh}, box_thresh={det_db_box_thresh}, unclip_ratio={det_db_unclip_ratio}")
        print(f"   Recognition threshold: score_thresh={rec_score_thresh}")
        print(f"   Runtime control: doc_ori={use_doc_orientation}, unwarping={use_doc_unwarping}, textline_ori={use_textline_orientation}")
        
        return npu_ocr
        
    except ImportError as e:
        error_msg = str(e)
        print(f"❌ NPU initialization failed: {error_msg}")
        raise HTTPException(status_code=503, detail=f"NPU initialization failed: {error_msg}")
    except Exception as e:
        print(f"❌ NPU initialization error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"NPU initialization error: {str(e)}")


class NPUOCRWrapper:
    """Wrapper for NPU PaddleOcr (sync) and AsyncPipelineOCR (async) to provide CPU-compatible interface"""
    
    def __init__(self, npu_ocr):
        self.npu_ocr = npu_ocr
        self.processed_imgs = []  # Store processed images for visualization
        # Check if this is sync (PaddleOcr) or async (AsyncPipelineOCR)
        self.is_sync = hasattr(npu_ocr, '__call__') and not hasattr(npu_ocr, 'process_batch')
    
    def predict(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
        **kwargs
    ) -> List[dict]:
        """
        Predict method compatible with CPU PaddleOCR interface
        Handles both single image and batch processing based on mode
        
        Args:
            imgs: Single image (np.ndarray) or list of images (List[np.ndarray])
        
        Returns:
            list: OCR results for each image in CPU-compatible format
                [
                    {'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]},
                    ...
                ]
        """
        # Normalize input to list
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
        
        results = []
        self.processed_imgs = []
        
        # Update runtime parameters (unified for both sync and async)
        params_updated = False
        
        # Update doc preprocessing flags
        if use_doc_orientation_classify is not None or use_doc_unwarping is not None:
            use_doc_preprocessing = use_doc_orientation_classify or use_doc_unwarping
            self.npu_ocr.use_doc_preprocessing = use_doc_preprocessing
            if self.npu_ocr.doc_preprocessing is not None:
                if use_doc_orientation_classify is not None:
                    self.npu_ocr.doc_preprocessing.use_doc_orientation = use_doc_orientation_classify
                if use_doc_unwarping is not None:
                    self.npu_ocr.doc_preprocessing.use_doc_unwarping = use_doc_unwarping
            params_updated = True
        
        # Update textline orientation
        if use_textline_orientation is not None:
            self.npu_ocr.use_textline_orientation = use_textline_orientation
            params_updated = True
        
        # Update detection parameters
        if text_det_thresh is not None:
            self.npu_ocr.det_db_thresh = text_det_thresh
            # Update detection node thresholds
            if hasattr(self.npu_ocr, 'detection_node'):
                self.npu_ocr.detection_node.det_postprocess.thresh = text_det_thresh
            params_updated = True
        if text_det_box_thresh is not None:
            self.npu_ocr.det_db_box_thresh = text_det_box_thresh
            if hasattr(self.npu_ocr, 'detection_node'):
                self.npu_ocr.detection_node.det_postprocess.box_thresh = text_det_box_thresh
            params_updated = True
        if text_det_unclip_ratio is not None:
            self.npu_ocr.det_db_unclip_ratio = text_det_unclip_ratio
            if hasattr(self.npu_ocr, 'detection_node'):
                self.npu_ocr.detection_node.det_postprocess.unclip_ratio = text_det_unclip_ratio
            params_updated = True
        
        # Update recognition parameters
        if text_rec_score_thresh is not None:
            self.npu_ocr.rec_score_thresh = text_rec_score_thresh
            if hasattr(self.npu_ocr, 'recognition_node'):
                self.npu_ocr.recognition_node.drop = text_rec_score_thresh
            params_updated = True
        
        # Print updated parameters
        if params_updated:
            mode_name = "Sync PaddleOcr" if self.is_sync else "Async AsyncPipelineOCR"
            print(f"🔧 NPU {mode_name} runtime params updated:")
            print(f"   doc_preprocessing: {self.npu_ocr.use_doc_preprocessing}")
            if self.npu_ocr.doc_preprocessing is not None:
                print(f"   doc_orientation: {self.npu_ocr.doc_preprocessing.use_doc_orientation}")
                print(f"   doc_unwarping: {self.npu_ocr.doc_preprocessing.use_doc_unwarping}")
            print(f"   textline_orientation: {self.npu_ocr.use_textline_orientation}")
            print(f"   det_thresh: {self.npu_ocr.det_db_thresh}, box_thresh: {self.npu_ocr.det_db_box_thresh}, unclip_ratio: {self.npu_ocr.det_db_unclip_ratio}")
            print(f"   rec_score_thresh: {self.npu_ocr.rec_score_thresh}")
        
        # Mode-specific processing
        if self.is_sync:
            # Sync mode: Sequential processing with PaddleOcr.__call__()
            for img in imgs:
                # PaddleOcr.__call__ returns: (boxes, crops, rec_results, processed_img, debug_data)
                boxes, crops, rec_results, processed_img, debug_data = self.npu_ocr(img)
                
                # Store preprocessed image for visualization
                self.processed_imgs.append(processed_img)
                
                # Convert NPU format to CPU-compatible format
                # Use bbox from rec_results to ensure alignment with filtered results
                texts = [r['text'] for r in rec_results]
                scores = [r['score'] for r in rec_results]
                polys = [r['bbox'] for r in rec_results]  # bbox from recognition results (filtered)
                
                results.append({
                    'rec_texts': texts,
                    'rec_scores': scores,
                    'rec_polys': polys,
                    'doc_preprocessor_res': {'output_img': processed_img}
                })
        else:
            # Async mode: Batch processing with AsyncPipelineOCR.process_batch()
            batch_results = self.npu_ocr.process_batch(imgs, timeout=60.0, pass_preprocessing=False)
            
            if not batch_results:
                batch_results = [{'rec_results': [], 'rec_texts': [], 'rec_scores': [], 'rec_polys': [], 'doc_preprocessor_res': {'output_img': None}} for _ in imgs]
            
            # AsyncPipelineOCR._format_results() returns CPU-compatible format
            for batch_result in batch_results:
                # Store processed image for visualization
                self.processed_imgs.append(batch_result.get('preprocessed_image', None))
                
                # Extract CPU-compatible format from batch_result
                results.append({
                    'rec_texts': batch_result.get('rec_texts', []),
                    'rec_scores': batch_result.get('rec_scores', []),
                    'rec_polys': batch_result.get('rec_polys', []),
                    'doc_preprocessor_res': batch_result.get('doc_preprocessor_res', {'output_img': None})
                })
        
        return results


# ============================================================================
# Document Preprocessing Models (Global Cache)
# ============================================================================

# Global document preprocessor instances
doc_orientation_classifier = None
doc_unwarping_model = None

def get_doc_orientation_classifier():
    """Get or create DocImgOrientationClassification instance (lazy loading)"""
    global doc_orientation_classifier
    if doc_orientation_classifier is None:
        try:
            # PaddleX automatically uses cached model if available
            doc_orientation_classifier = DocImgOrientationClassification()
            print("✅ DocImgOrientationClassification initialized (using cached model if available)")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize DocImgOrientationClassification: {e}")
            doc_orientation_classifier = None
    return doc_orientation_classifier

def get_doc_unwarping_model():
    """Get or create TextImageUnwarping instance (lazy loading)"""
    global doc_unwarping_model
    if doc_unwarping_model is None:
        try:
            # PaddleX automatically uses cached model if available
            doc_unwarping_model = TextImageUnwarping()
            print("✅ TextImageUnwarping initialized (using cached model if available)")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize TextImageUnwarping: {e}")
            doc_unwarping_model = None
    return doc_unwarping_model

# ============================================================================
# OCR Visualization Utilities (from PaddleOCR tools/infer/utility.py)
# ============================================================================

def get_font_path(font_name="simfang.ttf"):
    """
    Get font path with fallback logic
    Priority:
    1. FONT_PATH environment variable
    2. DEEPX_PATH/engine/fonts
    3. Relative path: deepx/engine/fonts
    4. System fonts
    """
    # 1. Check FONT_PATH environment variable
    font_path_env = os.getenv('FONT_PATH')
    if font_path_env:
        font_file = Path(font_path_env) / font_name
        if font_file.exists():
            return str(font_file)
    
    # 2. Check DEEPX_PATH/engine/fonts
    deepx_path_str = os.getenv('DEEPX_PATH')
    if deepx_path_str:
        font_file = Path(deepx_path_str) / 'engine' / 'fonts' / font_name
        if font_file.exists():
            return str(font_file)
    
    # 3. Check relative path
    font_file = Path(__file__).parent / 'deepx' / 'engine' / 'fonts' / font_name
    if font_file.exists():
        return str(font_file)
    
    # 4. Fallback to old location (for backward compatibility)
    font_file = Path(__file__).parent.parent.parent / 'doc' / 'fonts' / font_name
    if font_file.exists():
        return str(font_file)
    
    # 5. System font fallback
    system_font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if system_font.exists():
        return str(system_font)
    
    # Last resort: return default path (ImageFont.load_default will be used)
    return str(Path(__file__).parent / 'deepx' / 'engine' / 'fonts' / font_name)

def create_font(txt, sz, font_path=None):
    """Create font for text rendering"""
    # Get font path if not provided
    if font_path is None:
        font_path = get_font_path()
    
    font_size = int(sz[1] * 0.99)
    try:
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        return font
    
    if int(PIL.__version__.split(".")[0]) < 10:
        length = font.getsize(txt)[0]
    else:
        length = font.getlength(txt)

    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        try:
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        except:
            font = ImageFont.load_default()
    return font


def draw_box_txt_fine(img_size, box, txt, font_path=None):
    """Draw text in box with perspective transformation"""
    # Get font path if not provided
    if font_path is None:
        font_path = get_font_path()
    
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text


def draw_ocr_box_txt(
    image,
    boxes,
    txts=None,
    scores=None,
    drop_score=0.5,
    font_path=None,
):
    """
    Visualize OCR results with boxes and text
    Returns: numpy array with visualization (width * 2, height) showing original and text overlay
    """
    # Get font path if not provided
    if font_path is None:
        font_path = get_font_path()
    
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        # Copy text (non-white pixels) to img_right
        mask = img_right_text < 255
        img_right[mask] = img_right_text[mask]
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)

# Pydantic models for request/response

# Baidu AI Studio compatible request model
class BaiduOCRRequest(BaseModel):
    """Baidu AI Studio OCR API compatible request"""
    file: str = Field(..., description="Base64 encoded file content")
    fileType: int = Field(1, description="File type: 0=PDF, 1=Image")
    
    # Optional preprocessing parameters
    useDocOrientationClassify: Optional[bool] = Field(False, description="Use document orientation classification")
    useDocUnwarping: Optional[bool] = Field(False, description="Use document unwarping/distortion correction")
    useTextlineOrientation: Optional[bool] = Field(False, description="Use text line orientation classification")
    
    # Detection parameters
    textDetLimitSideLen: Optional[int] = Field(64, description="Detection image side length limit")
    textDetLimitType: Optional[str] = Field("min", description="Limit type: 'min' or 'max'")
    textDetThresh: Optional[float] = Field(0.3, description="Text detection threshold")
    textDetBoxThresh: Optional[float] = Field(0.6, description="Text detection box threshold")
    textDetUnclipRatio: Optional[float] = Field(1.5, description="Text detection unclip ratio")
    
    # Recognition parameters
    textRecScoreThresh: Optional[float] = Field(0.0, description="Text recognition score threshold")
    
    # Visualization
    visualize: Optional[bool] = Field(False, description="Return visualization images")
    
    # DEEPX NPU support
    deepx: Optional[bool] = Field(False, description="Use DEEPX NPU for inference (default: false, uses CPU)")
    sync: Optional[bool] = Field(False, description="Use sync NPU PaddleOCR instead of async pipeline (default: false, uses AsyncPipelineOCR)")

# Original request models
class OCRRequest(BaseModel):
    """OCR request with image URL or base64 encoded image(s)"""
    url: Optional[str] = Field(None, description="Image URL")
    image: Optional[str] = Field(None, description="Base64 encoded image (single)")
    images: Optional[List[str]] = Field(None, description="Base64 encoded images (array) - hubserving format")
    deepx: Optional[bool] = Field(False, description="Use DEEPX NPU for inference (default: false, uses CPU)")
    sync: Optional[bool] = Field(False, description="Use sync NPU PaddleOCR instead of async pipeline (default: false, uses AsyncPipelineOCR)")
    
    # Preprocessing options (consistent defaults across all endpoints)
    useDocOrientationClassify: Optional[bool] = Field(False, description="Use document orientation classification")
    useDocUnwarping: Optional[bool] = Field(False, description="Use document unwarping/distortion correction")
    useTextlineOrientation: Optional[bool] = Field(True, description="Use text line orientation classification")

class BatchOCRRequest(BaseModel):
    """Batch OCR request with multiple base64 encoded images"""
    images: List[str] = Field(..., description="List of base64 encoded images")
    deepx: Optional[bool] = Field(False, description="Use DEEPX NPU for inference (default: false, uses CPU)")
    sync: Optional[bool] = Field(False, description="Use sync NPU PaddleOCR instead of async pipeline (default: false, uses AsyncPipelineOCR)")
    
    # Preprocessing options (consistent defaults across all endpoints)
    useDocOrientationClassify: Optional[bool] = Field(False, description="Use document orientation classification")
    useDocUnwarping: Optional[bool] = Field(False, description="Use document unwarping/distortion correction")
    useTextlineOrientation: Optional[bool] = Field(True, description="Use text line orientation classification")

class OCRResult(BaseModel):
    """Single OCR detection result"""
    bbox: List[List[float]] = Field(..., description="Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    text: str = Field(..., description="Recognized text")
    confidence: float = Field(..., description="Confidence score (0-1)")

# Baidu AI Studio compatible response models
class BaiduOCRResult(BaseModel):
    """Single OCR result in Baidu format"""
    bbox: List[List[float]] = Field(..., description="Bounding box")
    text: str = Field(..., description="Recognized text")
    score: float = Field(..., description="Confidence score")

class BaiduPageResult(BaseModel):
    """OCR result for a single page/image"""
    prunedResult: Dict[str, Any] = Field(..., description="OCR results")
    ocrImage: Optional[str] = Field(None, description="OCR visualization image (base64)")
    docPreprocessingImage: Optional[str] = Field(None, description="Preprocessing image (base64)")
    inputImage: Optional[str] = Field(None, description="Input image (base64)")

class BaiduOCRResponse(BaseModel):
    """Baidu AI Studio compatible response"""
    logId: str = Field(..., description="Request UUID")
    errorCode: int = Field(0, description="Error code (0 = success)")
    errorMsg: str = Field("Success", description="Error message")
    result: Dict[str, Any] = Field(..., description="OCR results")

class OCRResponse(BaseModel):
    """OCR response"""
    success: bool = Field(..., description="Whether the request was successful")
    results: List[OCRResult] = Field(..., description="List of OCR results")

class BatchOCRResponse(BaseModel):
    """Batch OCR response"""
    success: bool = Field(..., description="Whether the request was successful")
    results: List[List[OCRResult]] = Field(..., description="List of OCR results for each image")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")

class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    traceback: Optional[str] = Field(None, description="Error traceback (if available)")

# Initialize FastAPI app
app = FastAPI(
    title="PaddleOCR Service",
    description="REST API for OCR inference using PP-OCRv5 with Baidu AI Studio compatibility",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global OCR instance cache
ocr_instances = {}

def get_ocr_instance(
    use_textline_orientation: bool = False,
    det_limit_side_len: int = 64,
    det_limit_type: str = "min",
    det_db_thresh: float = 0.3,
    det_db_box_thresh: float = 0.6,
    det_db_unclip_ratio: float = 1.5,
    rec_score_thresh: float = 0.0
) -> PaddleOCR:
    """
    Get or create PaddleOCR instance with specified parameters
    Uses caching to reuse instances with same parameters
    """
    use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'
    device = 'gpu:0' if use_gpu else 'cpu'
    
    # Get model paths based on USE_MOBILE environment variable
    model_paths = get_model_paths()
    
    # Create cache key from parameters (including model paths)
    cache_key = (
        use_textline_orientation,
        det_limit_side_len,
        det_limit_type,
        det_db_thresh,
        det_db_box_thresh,
        det_db_unclip_ratio,
        rec_score_thresh,
        device,
        model_paths['det_model_name'],
        model_paths['rec_model_name']
    )
    
    if cache_key not in ocr_instances:
        print(f"🔧 Initializing PaddleOCR with {model_paths['model_type']} models...")
        print(f"   Detection model name: {model_paths['det_model_name']}")
        print(f"   Detection model dir: {model_paths['det_model_dir']}")
        print(f"   Recognition model name: {model_paths['rec_model_name']}")
        print(f"   Recognition model dir: {model_paths['rec_model_dir']}")
        
        # Build PaddleOCR init parameters
        ocr_params = {
            'use_textline_orientation': use_textline_orientation,
            'text_det_limit_side_len': det_limit_side_len,
            'text_det_limit_type': det_limit_type,
            'text_det_thresh': det_db_thresh,
            'text_det_box_thresh': det_db_box_thresh,
            'text_det_unclip_ratio': det_db_unclip_ratio,
            'lang': 'ch',
            'device': device
        }
        
        # Add model names and directories if they exist
        if model_paths['det_model_name']:
            ocr_params['text_detection_model_name'] = model_paths['det_model_name']
        if model_paths['det_model_dir']:
            ocr_params['text_detection_model_dir'] = model_paths['det_model_dir']
        if model_paths['rec_model_name']:
            ocr_params['text_recognition_model_name'] = model_paths['rec_model_name']
        if model_paths['rec_model_dir']:
            ocr_params['text_recognition_model_dir'] = model_paths['rec_model_dir']
        
        # Add textline orientation model if it exists and is enabled
        if use_textline_orientation:
            models_dir = Path.home() / '.paddlex' / 'official_models'
            textline_ori_model_dir = models_dir / 'PP-LCNet_x1_0_textline_ori'
            if textline_ori_model_dir.exists():
                ocr_params['textline_orientation_model_name'] = 'PP-LCNet_x1_0_textline_ori'
                ocr_params['textline_orientation_model_dir'] = str(textline_ori_model_dir)
                print(f"   Using pre-downloaded textline orientation model: {textline_ori_model_dir}")
        
        ocr_instances[cache_key] = PaddleOCR(**ocr_params)
        print(f"✅ PaddleOCR initialized successfully with {model_paths['model_type']} models")
    
    return ocr_instances[cache_key]

# Default OCR instance (lazy loading)
ocr = None

def get_ocr_engine(
    deepx: bool = False,
    sync: bool = False,
    use_doc_orientation: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    det_limit_side_len: int = 64,
    det_limit_type: str = "min",
    det_db_thresh: float = 0.3,
    det_db_box_thresh: float = 0.6,
    det_db_unclip_ratio: float = 1.5,
    rec_score_thresh: float = 0.0
):
    """
    Unified OCR engine getter for both CPU and NPU
    Returns engine instance with predict() method
    
    Args:
        deepx: Use NPU if True, CPU if False
        sync: Use sync NPU PaddleOCR (True) or async AsyncPipelineOCR (False)
        use_doc_orientation: Enable document orientation classification
        use_doc_unwarping: Enable document unwarping
        use_textline_orientation: Enable textline orientation classification
        det_limit_side_len: Detection side length limit (CPU only)
        det_limit_type: Detection limit type (CPU only)
        det_db_thresh: Detection threshold
        det_db_box_thresh: Detection box threshold
        det_db_unclip_ratio: Detection unclip ratio
        rec_score_thresh: Recognition score threshold
    
    Returns:
        OCR engine with predict(img) method
    """
    if deepx:
        # NPU path - return wrapped NPU OCR instance
        npu_ocr = get_npu_ocr_instance(
            sync=sync,
            use_doc_orientation=use_doc_orientation,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            rec_score_thresh=rec_score_thresh
        )
        return NPUOCRWrapper(npu_ocr)
    else:
        # CPU path
        return get_ocr_instance(
            use_textline_orientation=use_textline_orientation,
            det_limit_side_len=det_limit_side_len,
            det_limit_type=det_limit_type,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            rec_score_thresh=rec_score_thresh
        )

def process_images_with_ocr(
    imgs: Union[np.ndarray, List[np.ndarray]],
    deepx: bool = False,
    sync: bool = False,
    use_doc_orientation: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    det_limit_side_len: int = 64,
    det_limit_type: str = "min",
    det_db_thresh: float = 0.3,
    det_db_box_thresh: float = 0.6,
    det_db_unclip_ratio: float = 1.5,
    rec_score_thresh: float = 0.0
) -> List[dict]:
    """
    Unified OCR processing for both CPU and NPU
    Handles single image or batch of images
    Includes preprocessing (doc orientation, unwarping) and OCR inference
    
    Args:
        imgs: Single image (np.ndarray) or list of images (List[np.ndarray])
        deepx: Use NPU if True, CPU if False
        sync: Use sync NPU PaddleOCR (True) or async AsyncPipelineOCR (False)
        use_doc_orientation: Enable document orientation classification
        use_doc_unwarping: Enable document unwarping
        use_textline_orientation: Enable textline orientation classification
        det_limit_side_len: Detection side length limit (CPU only)
        det_limit_type: Detection limit type (CPU only)
        det_db_thresh: Detection threshold
        det_db_box_thresh: Detection box threshold
        det_db_unclip_ratio: Detection unclip ratio
        rec_score_thresh: Recognition score threshold
    
    Returns:
        List[dict]: OCR results for each image with doc_preprocessor_res
            [{
                'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...],
                'doc_preprocessor_res': {'output_img': np.ndarray},
                ...
            }, ...]
    """
    # Normalize input to list
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    
    # Get OCR engine (unified for CPU/NPU)
    ocr_engine = get_ocr_engine(
        deepx=deepx,
        sync=sync,
        use_doc_orientation=use_doc_orientation,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        det_limit_side_len=det_limit_side_len,
        det_limit_type=det_limit_type,
        det_db_thresh=det_db_thresh,
        det_db_box_thresh=det_db_box_thresh,
        det_db_unclip_ratio=det_db_unclip_ratio,
        rec_score_thresh=rec_score_thresh
    )
    
    # Run OCR inference
    # NPU: NPU engine handles preprocessing internally
    backend = "NPU" if deepx else "CPU"
    mode = "sync" if sync else "async"
    print(f"🚀 Using {backend} ({mode}) for inference on {len(imgs)} image(s)")
    
    # Unified CPU/NPU inference - both return compatible format
    # CPU: _OCRPipeline handles doc_orientation, doc_unwarping internally
    # NPU: NPUOCRWrapper handles preprocessing and returns compatible format
    results = list(ocr_engine.predict(
        imgs,
        use_doc_orientation_classify=use_doc_orientation,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        text_det_limit_side_len=det_limit_side_len,
        text_det_limit_type=det_limit_type,
        text_det_thresh=det_db_thresh,
        text_det_box_thresh=det_db_box_thresh,
        text_det_unclip_ratio=det_db_unclip_ratio,
        text_rec_score_thresh=rec_score_thresh
    ))
    
    # Both CPU and NPU now return: {rec_texts, rec_scores, rec_polys, doc_preprocessor_res, ...}
    
    print(f"✅ {backend} OCR completed")
    
    return results

def process_with_doc_orientation(img: np.ndarray, use_deepx: bool = False) -> np.ndarray:
    """
    Process image with document orientation classification
    Uses PP-LCNet_x1_0_doc_ori model (CPU) or doc_ori_fixed.dxnn (NPU)
    
    Args:
        img: Input image as numpy array (BGR format)
        use_deepx: Use NPU model if True, CPU model if False
    
    Returns:
        np.ndarray: Rotated image if orientation is detected
    """
    if use_deepx:
        # NPU path - use DXNN model directly
        return process_with_doc_orientation_npu(img)
    
    # CPU path - use PaddleX model
    classifier = get_doc_orientation_classifier()
    if classifier is None:
        print("⚠️ DocImgOrientationClassification not available, returning original image")
        return img
    
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = img
        
        print(f"🔍 Running doc orientation classification on image size: {img_pil.size}")
        
        # Save to temporary file (PaddleX might require file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            img_pil.save(tmp_path)
        
        try:
            # Run document orientation classification
            results = classifier.predict(tmp_path)
            
            print(f"🔍 Doc orientation results: {type(results)}, len={len(results) if results else 0}")
            if results and len(results) > 0:
                result = results[0]
                print(f"🔍 Result type: {type(result)}")
                print(f"🔍 Result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                print(f"🔍 Result content: {result}")
                
                # Extract processed image from result
                if isinstance(result, dict) and 'img' in result:
                    processed_img = result['img']
                    # Convert back to numpy array if it's a PIL Image
                    if isinstance(processed_img, Image.Image):
                        processed_img = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2BGR)
                    print(f"✅ Orientation processed: {img.shape} -> {processed_img.shape}")
                    return processed_img
                elif isinstance(result, dict) and 'label_names' in result:
                    # Rotate based on classification result
                    label = result['label_names'][0] if result['label_names'] else '0'
                    print(f"🔍 Detected orientation: {label}")
                    
                    # Rotate image based on detected orientation
                    if '90' in str(label):
                        img_pil = img_pil.rotate(-90, expand=True)
                        print("🔄 Rotated image by -90 degrees")
                    elif '180' in str(label):
                        img_pil = img_pil.rotate(-180, expand=True)
                        print("🔄 Rotated image by -180 degrees")
                    elif '270' in str(label):
                        img_pil = img_pil.rotate(-270, expand=True)
                        print("🔄 Rotated image by -270 degrees")
                    else:
                        print("🔄 No rotation needed (0 degrees)")
                    
                    # Convert back to numpy
                    processed_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    return processed_img
            
            print("⚠️ Could not extract processed image, returning original")
            return img
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        print(f"⚠️ Error in doc orientation classification: {e}")
        import traceback
        traceback.print_exc()
        return img

def process_with_doc_unwarping(img: np.ndarray, use_deepx: bool = False) -> np.ndarray:
    """
    Process image with document unwarping using UVDoc model
    
    UVDoc (TextImageUnwarping) corrects document distortion (curving, warping)
    Uses UVDoc (CPU) or UVDoc_pruned_p3.dxnn (NPU)
    
    Args:
        img: Input image as numpy array (BGR format)
        use_deepx: Use NPU model if True, CPU model if False
        
    Returns:
        np.ndarray: Unwarped image (BGR format, same size as input)
        
    Notes:
        - UVDoc is a pixel-to-pixel model (input size = output size)
        - Result dict contains keys: input_path, page_index, input_img, doctr_img
        - doctr_img is the corrected image (numpy array)
    """
    if use_deepx:
        # NPU path - use DXNN model directly
        return process_with_doc_unwarping_npu(img)
    
    # CPU path - use PaddleX model
    unwarp_model = get_doc_unwarping_model()
    if unwarp_model is None:
        print("⚠️ TextImageUnwarping not available, returning original image")
        return img
    
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = img
        
        # Save to temporary file (PaddleX requires file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            img_pil.save(tmp_path)
        
        try:
            # Run document unwarping
            results = unwarp_model.predict(tmp_path)
            
            if results and len(results) > 0:
                result = results[0]
                
                # Result is a dict with keys: input_path, page_index, input_img, doctr_img
                # Extract the corrected image from 'doctr_img' key
                if isinstance(result, dict) and 'doctr_img' in result:
                    processed_img = result['doctr_img']
                    
                    # Convert to numpy array if needed
                    if isinstance(processed_img, Image.Image):
                        # PIL Image -> numpy array (RGB)
                        processed_img = np.array(processed_img)
                        # RGB -> BGR for OpenCV compatibility
                        if processed_img.ndim == 3 and processed_img.shape[2] == 3:
                            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                    elif isinstance(processed_img, np.ndarray):
                        # Already numpy array - check if RGB->BGR conversion needed
                        # PaddleX typically outputs RGB format
                        if processed_img.ndim == 3 and processed_img.shape[2] == 3:
                            # Assume RGB format from PaddleX, convert to BGR
                            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                    
                    print(f"✅ Unwarping: {img.shape} -> {processed_img.shape}")
                    return processed_img
                else:
                    print(f"⚠️ Unexpected result format. Keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
            
            print("⚠️ Could not extract processed image, returning original")
            return img
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        print(f"⚠️ Error in document unwarping: {e}")
        import traceback
        traceback.print_exc()
        return img

def process_with_doc_orientation_npu(img: np.ndarray) -> np.ndarray:
    """
    Process image with document orientation classification using NPU (DXNN model)
    Uses doc_ori_fixed.dxnn model directly
    
    Args:
        img: Input image as numpy array (BGR format)
    
    Returns:
        np.ndarray: Rotated image if orientation is detected
    """
    try:
        # Import DEEPX modules
        from deepx.engine.paddleocr import DocumentOrientationNode
        
        # Load NPU models
        models = load_npu_models_once()
        doc_ori_model = models['doc_ori_model']
        
        if doc_ori_model is None:
            print("⚠️ NPU doc_ori model not available, returning original image")
            return img
        
        # Create doc orientation node
        doc_node = DocumentOrientationNode(model=doc_ori_model)
        
        # Run inference - returns ((angle, rotated_image), latency)
        print(f"🔍 Running NPU doc orientation on image size: {img.shape}")
        (angle, output_img), latency = doc_node(img)
        
        print(f"✅ NPU orientation processed: angle={angle}°, latency={latency:.3f}s, {img.shape} -> {output_img.shape}")
        return output_img
        
    except Exception as e:
        print(f"⚠️ Error in NPU doc orientation: {e}")
        import traceback
        traceback.print_exc()
        return img

def process_with_doc_unwarping_npu(img: np.ndarray) -> np.ndarray:
    """
    Process image with document unwarping using NPU (DXNN model)
    Uses UVDoc_pruned_p3.dxnn model directly
    
    Args:
        img: Input image as numpy array (BGR format)
        
    Returns:
        np.ndarray: Unwarped image (BGR format)
    """
    try:
        # Import DEEPX modules
        from deepx.engine.paddleocr import DocumentUnwarpingNode
        
        # Load NPU models
        models = load_npu_models_once()
        doc_unwarping_model = models['doc_unwarping_model']
        
        if doc_unwarping_model is None:
            print("⚠️ NPU UVDoc model not available, returning original image")
            return img
        
        # Create doc unwarping node
        unwarp_node = DocumentUnwarpingNode(model=doc_unwarping_model)
        
        # Run inference - returns (unwarped_image, latency)
        print(f"🔍 Running NPU doc unwarping on image size: {img.shape}")
        output_img, latency = unwarp_node(img)
        
        print(f"✅ NPU unwarping processed: latency={latency:.3f}s, {img.shape} -> {output_img.shape}")
        return output_img
        
    except Exception as e:
        print(f"⚠️ Error in NPU doc unwarping: {e}")
        import traceback
        traceback.print_exc()
        return img

def create_visualization(preprocessing_img: np.ndarray, ocr_results: List[Dict]) -> str:
    """
    Create OCR visualization image with bounding boxes and text
    Left side: Preprocessed image with OCR boxes
    Right side: Text overlay on white background
    
    Args:
        preprocessing_img: Preprocessed image (OCR was performed on this image)
        ocr_results: OCR detection and recognition results
    
    Returns:
        Base64 encoded visualization image (2x width format)
        
    Note:
        OCR boxes are in preprocessing_img coordinates for both CPU and NPU
        because OCR is performed on the preprocessed image.
    """
    # Get font path using unified function
    font_path = get_font_path()
    
    # Convert preprocessing image to PIL Image for visualization
    if isinstance(preprocessing_img, np.ndarray):
        preprocessing_pil = Image.fromarray(cv2.cvtColor(preprocessing_img, cv2.COLOR_BGR2RGB))
    else:
        preprocessing_pil = preprocessing_img
    
    # Extract boxes, texts, scores from results
    boxes = []
    txts = []
    scores = []
    
    for result in ocr_results:
        boxes.append(result['bbox'])
        txts.append(result['text'])
        scores.append(result['confidence'])
    
    # Use draw_ocr_box_txt function if we have results
    if len(boxes) > 0:
        try:
            vis_img = draw_ocr_box_txt(
                preprocessing_pil,
                boxes,
                txts,
                scores,
                drop_score=0.0,  # Show all results
                font_path=font_path
            )
        except Exception as e:
            print(f"Error in draw_ocr_box_txt: {e}")
            # Fallback to simple visualization
            vis_img = np.array(preprocessing_pil)
            for result in ocr_results:
                bbox = result['bbox']
                pts = np.array(bbox, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
    else:
        # No results, just return preprocessing image
        vis_img = np.array(preprocessing_pil)
    
    # Convert to base64
    if isinstance(vis_img, Image.Image):
        vis_img = np.array(vis_img)
    
    # Ensure BGR format for cv2.imencode
    if len(vis_img.shape) == 3 and vis_img.shape[2] == 3:
        # Check if RGB, convert to BGR
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode('.jpg', vis_img)
    img_base64 = base64.b64encode(buffer).decode('ascii')
    
    return img_base64

def filter_results_by_score(results: List[Dict], threshold: float) -> List[Dict]:
    """
    Filter OCR results by confidence score threshold
    """
    return [r for r in results if r['confidence'] >= threshold]

def format_ocr_result(result: dict) -> List[OCRResult]:
    """
    Format OCR result from unified format to OCRResult list
    
    Args:
        result: OCR result in unified format {'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}
    
    Returns:
        List of OCRResult objects
    """
    formatted_results = []
    
    if result and isinstance(result, dict):
        texts = result.get('rec_texts', [])
        scores = result.get('rec_scores', [])
        polys = result.get('rec_polys', [])
        
        for i in range(len(texts)):
            bbox = polys[i].tolist() if i < len(polys) and isinstance(polys[i], np.ndarray) else (polys[i] if i < len(polys) else [])
            score = scores[i] if i < len(scores) else 0.0
            
            formatted_results.append(OCRResult(
                bbox=bbox,
                text=texts[i],
                confidence=float(score)
            ))
    
    return formatted_results

@app.get('/health', response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns service status
    """
    return HealthResponse(status='healthy')

@app.post('/api/v1/ocr', response_model=BaiduOCRResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, tags=["Baidu Compatible"])
async def baidu_ocr(request: BaiduOCRRequest):
    """
    Baidu AI Studio OCR API compatible endpoint
    
    Supports all Baidu OCR parameters including:
    - useDocOrientationClassify: Document orientation correction
    - useDocUnwarping: Image distortion correction  
    - useTextlineOrientation: Text line orientation correction
    - Detection parameters: textDetLimitSideLen, textDetThresh, etc.
    - Recognition parameters: textRecScoreThresh
    - visualize: Return visualization images
    
    Returns Baidu-compatible response format
    """
    try:
        # Generate request ID
        log_id = str(uuid.uuid4())
        
        # Decode base64 file
        try:
            file_data = base64.b64decode(request.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {str(e)}")
        
        # Check file type (currently only support images, not PDF)
        if request.fileType == 0:
            raise HTTPException(status_code=400, detail="PDF files not yet supported. Please set fileType=1 for images")
        
        # Load image
        img = Image.open(BytesIO(file_data))
        
        # Convert to RGB if needed (handle grayscale, RGBA, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_np = np.array(img)
        
        # Run OCR inference with preprocessing (unified CPU/NPU flow)
        # Preprocessing is handled inside process_images_with_ocr:
        # - CPU: Uses PaddleX models (process_with_doc_orientation, process_with_doc_unwarping)
        # - NPU: Handled internally in NPU PaddleOcr.__call__()
        backend = "NPU" if request.deepx else "CPU"
        print(f"🔧 {backend} OCR parameters:")
        print(f"   useDocOrientationClassify: {request.useDocOrientationClassify}")
        print(f"   useDocUnwarping: {request.useDocUnwarping}")
        print(f"   useTextlineOrientation: {request.useTextlineOrientation}")
        
        # Run OCR inference using unified function (preprocessing handled inside)
        results = process_images_with_ocr(
            imgs=[img_np],
            deepx=request.deepx,
            sync=request.sync,
            use_doc_orientation=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_textline_orientation=request.useTextlineOrientation,
            det_limit_side_len=request.textDetLimitSideLen,
            det_limit_type=request.textDetLimitType,
            det_db_thresh=request.textDetThresh,
            det_db_box_thresh=request.textDetBoxThresh,
            det_db_unclip_ratio=request.textDetUnclipRatio,
            rec_score_thresh=request.textRecScoreThresh
        )
        
        # Get preprocessed image for visualization from doc_preprocessor_res
        preprocessing_img = results[0]['doc_preprocessor_res']['output_img']
        
        # Unified result processing for both CPU and NPU
        ocr_results = []
        if results and isinstance(results, list) and len(results) > 0:
            first_result = results[0]
            if isinstance(first_result, dict):
                texts = first_result.get('rec_texts', [])
                scores = first_result.get('rec_scores', [])
                polys = first_result.get('rec_polys', [])
                
                for i in range(len(texts)):
                    score = float(scores[i]) if i < len(scores) else 0.0
                    
                    # Apply score threshold
                    if score >= request.textRecScoreThresh:
                        bbox = polys[i].tolist() if i < len(polys) and isinstance(polys[i], np.ndarray) else (polys[i] if i < len(polys) else [])

                        ocr_results.append({
                            'bbox': bbox,
                            'text': texts[i],
                            'confidence': score,
                            'score': score
                        })
        
        print(f"📊 Final results after filtering: {len(ocr_results)} (threshold: {request.textRecScoreThresh})")
        
        # Create pruned result
        pruned_result = {
            'dt_polys': [r['bbox'] for r in ocr_results],
            'rec_texts': [r['text'] for r in ocr_results],
            'rec_scores': [r['score'] for r in ocr_results]
        }
        
        # Create page result
        page_result = {
            'prunedResult': pruned_result,
            'ocrImage': None,
            'docPreprocessingImage': None,
            'inputImage': None
        }
        
        # Add visualization images if requested
        if request.visualize:
            # Draw OCR boxes on the preprocessed image
            # Boxes are in preprocessing_img coordinates for both CPU and NPU
            
            # # Debug: Save OCR results as JSON
            # debug_json_path = f'test_outputs/ocr_results_{log_id}_{request.sync if "sync" else "async"}.json'
            # with open(debug_json_path, 'w', encoding='utf-8') as f:
            #     json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            # print(f"🐛 Debug: OCR results saved to {debug_json_path}")
                        
            page_result['ocrImage'] = create_visualization(preprocessing_img, ocr_results)
            
            # Preprocessing visualization (show the preprocessed image without boxes)
            if request.useDocOrientationClassify or request.useDocUnwarping:
                _, buffer = cv2.imencode('.jpg', preprocessing_img)
                page_result['docPreprocessingImage'] = base64.b64encode(buffer).decode('ascii')
            
            # Original input image (before any preprocessing)
            _, buffer = cv2.imencode('.jpg', img_np)
            page_result['inputImage'] = base64.b64encode(buffer).decode('ascii')
        
        # Build response
        response = BaiduOCRResponse(
            logId=log_id,
            errorCode=0,
            errorMsg="Success",
            result={
                'ocrResults': [page_result],
                'dataInfo': {
                    'fileType': request.fileType,
                    'imageCount': 1
                }
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error in Baidu OCR: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={
                'logId': str(uuid.uuid4()),
                'errorCode': 500,
                'errorMsg': str(e)
            }
        )

@app.post('/fastapi/ocr', responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, tags=["FastAPI OCR"])
async def ocr_image(request: OCRRequest):
    """
    FastAPI OCR endpoint for single/batch image using JSON body
    
    Accepts JSON body with `url` (image URL) or `image` (base64 string) or `images` (array)
    
    Returns OCR results with bounding boxes, text, and confidence scores
    - Single image (url/image): OCRResponse
    - Multiple images (images array): BatchOCRResponse
    
    Processing scenarios:
    - NPU + multiple images + async: AsyncPipelineOCR batch processing
    - NPU + (single image OR sync): Sequential PaddleOcr processing
    - CPU: Always sequential processing
    
    Note: OCR engine is selected dynamically based on request.deepx parameter
    No pre-initialization needed - get_ocr_engine() handles lazy loading
    """
    try:
        # Extract images from request
        imgs = []
        
        if request.image:
            # Single base64 image
            image_data = base64.b64decode(request.image)
            img = Image.open(BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            imgs.append(np.array(img))
        elif request.images:
            # Multiple base64 images
            for img_b64 in request.images:
                image_data = base64.b64decode(img_b64)
                img_pil = Image.open(BytesIO(image_data))
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                imgs.append(np.array(img_pil))
        elif request.url:
            # Image URL - download from web
            import requests as req
            response = req.get(request.url, timeout=10)
            response.raise_for_status()
            img_pil = Image.open(BytesIO(response.content))
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            imgs.append(np.array(img_pil))
        else:
            raise HTTPException(status_code=400, detail="No image provided (url, image, or images required in JSON body)")
        
        is_batch = len(imgs) > 1
        
        # Process images - all batch/single logic handled inside predict()
        results = process_images_with_ocr(
            imgs,
            deepx=request.deepx,
            sync=request.sync,
            use_doc_orientation=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_textline_orientation=request.useTextlineOrientation
        )
        
        # Format resultsformat_ocr_result
        all_results = [format_ocr_result(result) for result in results]
        
        # Return appropriate response type
        if is_batch:
            return BatchOCRResponse(success=True, results=all_results)
        else:
            return OCRResponse(success=True, results=all_results[0])
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error in OCR: {error_msg}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                error=str(e),
                traceback=error_msg
            ).dict()
        )

@app.post('/fastapi/ocr/upload', response_model=OCRResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, tags=["FastAPI OCR"])
async def ocr_upload(file: UploadFile = File(...), deepx: bool = Form(False), sync: bool = Form(False)):
    """
    FastAPI OCR endpoint for file upload
    
    Accepts file upload via multipart/form-data
    
    Args:
        file: Image file
        deepx: Use DEEPX NPU for inference (default: false, uses CPU)
        sync: Use sync NPU PaddleOCR instead of async pipeline (default: false)
    
    Returns OCR results with bounding boxes, text, and confidence scores
    
    Note: OCR engine is selected dynamically based on deepx parameter
    No pre-initialization needed - get_ocr_engine() handles lazy loading
    """
    try:
        # File upload
        img = Image.open(file.file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        
        # Run OCR inference (unified interface)
        results = process_images_with_ocr(
            img,
            deepx=deepx,
            sync=sync
        )
        
        # Format results using format_ocr_result
        formatted_results = []
        if results and len(results) > 0:
            formatted_results = format_ocr_result(results[0])
        
        return OCRResponse(success=True, results=formatted_results)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error in OCR upload: {error_msg}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                error=str(e),
                traceback=error_msg
            ).dict()
        )


@app.post('/predict/ocr_system', responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, tags=["OCR"])
async def predict_ocr_system(request: OCRRequest):
    """
    Hubserving-compatible OCR endpoint
    
    Supports both single image and batch processing:
    - Single: request.image or request.url → OCRResponse
    - Batch: request.images → BatchOCRResponse
    
    Processing modes (all handled inside predict()):
    - NPU + multiple images + async: AsyncPipelineOCR batch processing
    - NPU + (single image OR sync): Sequential PaddleOcr processing
    - CPU: Always sequential processing
    
    Note: OCR engine is selected dynamically based on request.deepx parameter
    No pre-initialization needed - get_ocr_engine() handles lazy loading
    """
    try:
        # Extract images from request
        imgs = []
        
        if request.image:
            # Single base64 image
            image_data = base64.b64decode(request.image)
            img = Image.open(BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            imgs.append(np.array(img))
        elif request.images:
            # Multiple base64 images
            for img_b64 in request.images:
                image_data = base64.b64decode(img_b64)
                img_pil = Image.open(BytesIO(image_data))
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                imgs.append(np.array(img_pil))
        elif request.url:
            # Image URL - download from web
            import requests as req
            response = req.get(request.url, timeout=10)
            response.raise_for_status()
            img_pil = Image.open(BytesIO(response.content))
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            imgs.append(np.array(img_pil))
        else:
            raise HTTPException(status_code=400, detail="No image provided (url, image, or images required in JSON body)")
        
        # Detect if this is a batch request based on input type
        is_batch = request.images is not None and len(request.images) > 0
        
        # Process images - all batch/single logic handled inside predict()
        results = process_images_with_ocr(
            imgs,
            deepx=request.deepx,
            sync=request.sync,
            use_doc_orientation=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_textline_orientation=request.useTextlineOrientation
        )
        
        # Format results
        all_results = [format_ocr_result(result) for result in results]
        
        # Return appropriate response type
        if is_batch:
            return BatchOCRResponse(success=True, results=all_results)
        else:
            return OCRResponse(success=True, results=all_results[0])
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error in batch OCR: {error_msg}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                error=str(e),
                traceback=error_msg
            ).dict()
        )

@app.post('/fastapi/batch_ocr', response_model=BatchOCRResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, tags=["FastAPI OCR"])
async def batch_ocr(request: BatchOCRRequest):
    """
    FastAPI Batch OCR endpoint
    
    Accepts array of base64 encoded images
    
    Returns OCR results for each image
    
    Note: OCR engine is selected dynamically based on request.deepx parameter
    No pre-initialization needed - get_ocr_engine() handles lazy loading
    """
    try:
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        # Decode all images
        imgs = []
        for img_b64 in request.images:
            image_data = base64.b64decode(img_b64)
            img = Image.open(BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            imgs.append(np.array(img))
        
        # Process all images using unified interface
        results = process_images_with_ocr(
            imgs,
            deepx=request.deepx,
            sync=request.sync,
            use_doc_orientation=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_textline_orientation=request.useTextlineOrientation
        )
        
        # Format results
        all_results = [format_ocr_result(result) for result in results]
        
        return BatchOCRResponse(success=True, results=all_results)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error in batch OCR: {error_msg}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                error=str(e),
                traceback=error_msg
            ).dict()
        )

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    host = os.getenv('HOST', '0.0.0.0')
    uvicorn.run(app, host=host, port=port, log_level="info")
