import os
import sys
import time
import numpy as np
import cv2
import torch
from typing import List, Tuple, Dict, Optional
from dx_engine import InferenceEngine as IE

engine_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(engine_path)

from preprocessing import PreProcessingCompose
from .utils import torch_to_numpy

# Import Node base class from paddleocr
from .paddleocr import Node
