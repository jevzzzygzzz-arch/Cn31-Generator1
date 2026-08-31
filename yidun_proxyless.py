# Complete Enhanced yidun_proxyless.py - All functions included
import json
import os
import random
import re
import string
import time
import warnings
import math
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import execjs
from loguru import logger
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import queue
from functools import lru_cache
from fake_useragent import UserAgent
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
warnings.filterwarnings("ignore", message=".*SIFT_create.*deprecated.*")

# ============ CONFIGURATION ============
DEBUG = False
MAX_RETRIES = 3
BATCH_SIZE = 5  # Tokens per batch
CACHE_SIZE = 1000

# Thread pool configuration
CPU_CORES = os.cpu_count() or 4
NUM_WORKERS = min(8, CPU_CORES * 2)  # Increased for better throughput
PREFETCH_WORKERS = 2

# Timeouts
CONNECTION_TIMEOUT = 3
READ_TIMEOUT = 8
TOTAL_TIMEOUT = 15

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = 'cuda' if USE_CUDA else 'cpu'

TOKEN_SERVER_URL = os.environ.get('TOKEN_SERVER_URL', 'https://cn31-antrax-solver-production.up.railway.app')
TOKEN_SAVE_ENDPOINT = f"{TOKEN_SERVER_URL}/api/save-token"

# ============ TOKEN HANDLING ============
file_lock = threading.Lock()
TOKEN_OUTPUT_FILE = os.path.join(DIR_PATH, 'validated_tokens.txt')

class TokenBuffer:
    """Thread-safe token buffer for batch processing"""
    def __init__(self, max_size=50):
        self.buffer = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()
        self.batch_queue = queue.Queue()
        self.total_tokens = 0
        self.token_count = 0
        self._stop = False
        
    def add_token(self, token):
        """Add token to buffer"""
        try:
            self.buffer.put_nowait(token)
            with self.lock:
                self.total_tokens += 1
                self.token_count += 1
            return True
        except queue.Full:
            return False
            
    def get_batch(self, batch_size=10):
        """Get batch of tokens"""
        tokens = []
        try:
            while len(tokens) < batch_size:
                token = self.buffer.get_nowait()
                tokens.append(token)
        except queue.Empty:
            pass
        return tokens
    
    def get_stats(self):
        with self.lock:
            return {
                'total': self.total_tokens,
                'buffer_size': self.buffer.qsize(),
                'token_count': self.token_count
            }

# Global token buffer
TOKEN_BUFFER = TokenBuffer(max_size=100)

def send_token_to_server_batch(tokens):
    """Send multiple tokens in one request"""
    if not tokens:
        return False
    try:
        payload = {"tokens": tokens, "batch": True}
        r = requests.post(TOKEN_SAVE_ENDPOINT, json=payload, timeout=5)
        return r.status_code in [200, 201]
    except:
        return False

def send_token_to_server(token):
    """Single token send with retry"""
    for attempt in range(2):
        try:
            payload = {"token": token}
            r = requests.post(TOKEN_SAVE_ENDPOINT, json=payload, timeout=3)
            if r.status_code in [200, 201]:
                return True
        except:
            time.sleep(0.1)
    return False

# ============ MODEL LOADING ============
_model_state = None
_model_lock = threading.Lock()
_model_loaded = False

def initialize_global_model():
    global _model_state, _model_loaded
    
    if _model_loaded:
        return _model_state
    
    with _model_lock:
        if _model_loaded:
            return _model_state
            
        model_path = os.path.join(DIR_PATH, 'net.pkl')
        if not os.path.exists(model_path):
            logger.error("Model file net.pkl not found")
            return None
            
        try:
            # Load with optimized settings
            state = torch.load(model_path, map_location=torch.device(DEVICE), weights_only=False)
            
            if 'net' in state:
                state['net'] = state['net'].to(DEVICE)
                state['net'].eval()
                
                # Optimize for inference
                if USE_CUDA:
                    state['net'] = state['net'].half()
                    # Warm up CUDA
                    dummy = torch.randn(1, 3, 416, 416).half().to(DEVICE)
                    with torch.no_grad():
                        state['net'](dummy)
                
                # Compile model for faster inference (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    try:
                        state['net'] = torch.compile(state['net'], mode='reduce-overhead')
                    except:
                        pass
                        
            _model_state = state
            _model_loaded = True
            logger.success(f"Model loaded on {DEVICE} with optimizations")
            return _model_state
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return None

def get_global_model():
    global _model_state, _model_loaded
    if not _model_loaded:
        return initialize_global_model()
    return _model_state

# ============ JS ENGINE ============
_js_cache = {}
_js_lock = threading.Lock()

@lru_cache(maxsize=10)
def get_compiled_js_cached(file_name):
    try:
        js_path = os.path.join(DIR_PATH, file_name)
        with open(js_path, 'r', encoding='utf-8') as f:
            js_code = f.read()
        
        # Pre-compile with optimizations
        ctx = execjs.compile(js_code)
        # Warm up JS engine
        try:
            ctx.call('get_cb')
        except:
            pass
        return ctx
    except:
        return None

def get_compiled_js(file_name):
    return get_compiled_js_cached(file_name)

# ============ SIFT DETECTOR ============
_sift_detector = None
_sift_lock = threading.Lock()

def get_sift_detector():
    global _sift_detector
    if _sift_detector is None:
        with _sift_lock:
            if _sift_detector is None:
                try:
                    # Use fewer features for speed
                    _sift_detector = cv2.SIFT_create(nfeatures=30, contrastThreshold=0.1)
                except AttributeError:
                    try:
                        _sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures=30, contrastThreshold=0.1)
                    except:
                        _sift_detector = cv2.ORB_create(nfeatures=30)
    return _sift_detector

# ============ CONFIGURATION ============
REFERER = "https://mtacc.mobilelegends.com/"
ID = "fef5c67c39074e9d845f4bf579cc07af"
FP_H = "mtacc.mobilelegends.com"

DUN163_DOMAINS = [
    "https://c.dun.163.com",
    "https://c.dun.163yun.com"
]

# ============ ORIGINAL FUNCTIONS (MISSING) ============

def emergency_fallback():
    """Emergency fallback for click points"""
    return [(80, 70), (160, 120), (240, 90)]

def safe_list_access(lst, index, default=None):
    """Safely access list element"""
    try:
        if lst is None or not isinstance(lst, (list, tuple)):
            return default
        if not (0 <= index < len(lst)):
            return default
        return lst[index]
    except:
        return default

def rotate_about_center(src, angle, scale=1.):
    """Rotate image about center"""
    try:
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LINEAR)
    except:
        return src

def parse_y_pred(ypred, anchors, class_types, islist=False, threshold=0.2, nms_threshold=0):
    """Parse YOLO predictions"""
    try:
        if not anchors or not class_types:
            return [] if islist else None

        ceillen = 5 + len(class_types)
        sigmoid = lambda x: 1/(1+math.exp(-x))
        infos = []

        for idx in range(min(len(anchors), 3)):
            try:
                tensor_idx = 4 + idx * ceillen
                if tensor_idx >= ypred.shape[3]:
                    continue

                if USE_CUDA:
                    a = ypred[:,:,:,tensor_idx].cpu().detach().numpy()
                else:
                    a = ypred[:,:,:,tensor_idx].detach().numpy()

                for ii, i in enumerate(a[0]):
                    for jj, j in enumerate(i):
                        infos.append((ii, jj, idx, sigmoid(j)))
            except:
                continue

        if not infos:
            return [] if islist else None

        infos = sorted(infos, key=lambda i: -i[3])

        def get_xyxy_clz_con_emergency(info):
            try:
                gap = 416/ypred.shape[1]
                x, y, idx, con = info

                if idx >= len(anchors):
                    return None

                gp = idx * ceillen

                if (gp + 5 + len(class_types)) > ypred.shape[3]:
                    return None

                contain = torch.sigmoid(ypred[0, x, y, gp+4])
                pred_xy = torch.sigmoid(ypred[0, x, y, gp+0:gp+2])
                pred_wh = ypred[0, x, y, gp+2:gp+4]
                pred_clz = ypred[0, x, y, gp+5:gp+5+len(class_types)]

                if USE_CUDA:
                    pred_xy = pred_xy.cpu().detach().numpy()
                    pred_wh = pred_wh.cpu().detach().numpy()
                    pred_clz = pred_clz.cpu().detach().numpy()
                else:
                    pred_xy = pred_xy.detach().numpy()
                    pred_wh = pred_wh.detach().numpy()
                    pred_clz = pred_clz.detach().numpy()

                exp = math.exp
                cx, cy = float(pred_xy[0]), float(pred_xy[1])
                rx, ry = (cx + x)*gap, (cy + y)*gap
                rw, rh = float(pred_wh[0]), float(pred_wh[1])
                rw, rh = exp(rw)*anchors[idx][0], exp(rh)*anchors[idx][1]
                clz_ = [float(x) for x in pred_clz]
                xx = rx - rw/2
                _x = rx + rw/2
                yy = ry - rh/2
                _y = ry + rh/2

                if USE_CUDA:
                    log_cons = torch.sigmoid(ypred[:,:,:,gp+4]).cpu().detach().numpy()
                else:
                    log_cons = torch.sigmoid(ypred[:,:,:,gp+4]).detach().numpy()

                log_cons = np.transpose(log_cons, (0, 2, 1))

                clz = 'unknown'
                if clz_:
                    max_val = max(clz_)
                    max_idx = clz_.index(max_val)
                    for key, value in class_types.items():
                        if value == max_idx:
                            clz = key
                            break

                return [xx, yy, _x, _y], clz, con, log_cons

            except:
                return None

        if islist:
            limited_infos = infos[:min(50, len(infos))]
            v = []
            for i in limited_infos:
                if i[3] > threshold:
                    result = get_xyxy_clz_con_emergency(i)
                    if result is not None:
                        v.append(result)
            return v
        else:
            if infos:
                return get_xyxy_clz_con_emergency(infos[0])
            return None

    except Exception as e:
        logger.error(f"Emergency: parse_y_pred failed: {e}")
        return [] if islist else None

class Mini(nn.Module):
    class ConvBN(nn.Module):
        def __init__(self, cin, cout, kernel_size=3, stride=1, padding=None):
            super().__init__()
            padding = (kernel_size - 1) // 2 if not padding else padding
            self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(cout, momentum=0.01)
            self.relu = nn.LeakyReLU(0.1, inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    def __init__(self, anchors, class_types, inchennel=3):
        super().__init__()
        self.oceil = len(anchors) * (5 + len(class_types))
        self.model = nn.Sequential(
            OrderedDict([
                ('ConvBN_0', self.ConvBN(inchennel, 32)),
                ('Pool_0', nn.MaxPool2d(2, 2)),
                ('ConvBN_1', self.ConvBN(32, 48)),
                ('Pool_1', nn.MaxPool2d(2, 2)),
                ('ConvBN_2', self.ConvBN(48, 64)),
                ('Pool_2', nn.MaxPool2d(2, 2)),
                ('ConvBN_3', self.ConvBN(64, 80)),
                ('Pool_3', nn.MaxPool2d(2, 2)),
                ('ConvBN_4', self.ConvBN(80, 96)),
                ('Pool_4', nn.MaxPool2d(2, 2)),
                ('ConvBN_5', self.ConvBN(96, 102)),
                ('ConvEND', nn.Conv2d(102, self.oceil, 1)),
            ])
        )

    def forward(self, x):
        return self.model(x).permute(0, 2, 3, 1)

def get_clz_rect_from_image(image_data, state):
    """Get class and rectangle from image"""
    try:
        if not state or 'net' not in state:
            return [], None

        net = state['net']
        anchors = state.get('anchors', [])
        class_types = state.get('class_types', {})

        if not anchors or not class_types:
            return [], None

        image_array = np.frombuffer(image_data, dtype=np.uint8)
        npimg = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if npimg is None:
            return [], None

        height, width = npimg.shape[:2]
        npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
        npimg = cv2.resize(npimg, (416, 416), interpolation=cv2.INTER_LINEAR)
        npimg_ = np.transpose(npimg, (2,1,0))

        with torch.no_grad():
            input_tensor = torch.FloatTensor(npimg_).unsqueeze(0).to(DEVICE)
            if USE_CUDA:
                input_tensor = input_tensor.half()

            y_pred = net(input_tensor)

        v = parse_y_pred(y_pred, anchors, class_types, islist=True, threshold=0.2, nms_threshold=0.4)
        ret = []

        for i in v:
            if len(i) >= 4:
                rect, clz, con, log_cons = i[0], i[1], i[2], i[3]
                rw, rh = width/416, height/416
                rect[0] = int(rect[0]*rw)
                rect[2] = int(rect[2]*rw)
                rect[1] = int(rect[1]*rh)
                rect[3] = int(rect[3]*rh)
                ret.append([clz, rect])

        return ret, npimg

    except Exception as e:
        logger.error(f"Emergency: get_clz_rect_from_image failed: {e}")
        return [], None

def get_cut_img(npimg, rects):
    """Cut image based on rectangles"""
    ret = []
    try:
        for item in rects:
            if len(item) >= 2:
                clz, rect = item[0], item[1]
                if len(rect) >= 4:
                    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(npimg.shape[1], x2), min(npimg.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        ret.append([clz, npimg[y1:y2,x1:x2,:], (x1,y1,x2,y2)])
    except:
        pass
    return ret

def get_match_lens_emergency(i1, i2):
    """Get match length using SIFT"""
    try:
        if i1.size == 0 or i2.size == 0:
            return 0

        i1 = cv2.resize(i1, (min(i1.shape[1]*4, 800), min(i1.shape[0]*4, 600)), interpolation=cv2.INTER_LINEAR)
        i2 = cv2.resize(i2, (min(i2.shape[1]*2, 400), min(i2.shape[0]*2, 300)), interpolation=cv2.INTER_LINEAR)

        sift = get_sift_detector()
        kp1, des1 = sift.detectAndCompute(i1, None)
        kp2, des2 = sift.detectAndCompute(i2, None)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = 0

        for match_pair in matches:
            if len(match_pair) >= 2:
                m, n = match_pair[0], match_pair[1]
                if m.distance <= 0.88 * n.distance:
                    good += 1

        return good

    except:
        return 0

def get_flag_rect_emergency(k12, cut_imgs, st):
    """Get flag rectangle emergency"""
    try:
        if len(k12) < 2:
            return []

        k1, k2 = k12[0], k12[1]
        r = []

        for item in cut_imgs:
            if len(item) >= 3:
                clz, npimg, rect = item[0], item[1], item[2]
                if clz == '1':
                    r1 = get_match_lens_emergency(k1, npimg)
                    r.append([r1, rect, st])
                elif clz == '2':
                    r2 = get_match_lens_emergency(k2, npimg)
                    r.append([r2, rect, st])

        return sorted(r, key=lambda i: i[0]) if r else []

    except:
        return []

def get_flags_rects_from_image(image_data, state):
    """Get flag rectangles from image"""
    try:
        if state is None:
            return None, None, None

        image_array = np.frombuffer(image_data, dtype=np.uint8)
        s = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if s is None or s.size == 0:
            return None, None, None

        height, width = s.shape[:2]

        if height < 200 or width < 84:
            return None, None, None

        try:
            end_height = min(height, s.shape[0])
            a = s[160:end_height, 0:min(22, width), :]
            b = s[160:end_height, 28:min(50, width), :]
            c = s[160:end_height, 56:min(78, width), :]

            if a.shape[0] < 40 or a.shape[1] < 20:
                return None, None, None

            a1 = a[40:min(60, a.shape[0]), :, :] if a.shape[0] > 40 else a
            a2 = a[0:min(20, a.shape[0]), :, :] if a.shape[0] > 0 else a
            b1 = b[40:min(60, b.shape[0]), :, :] if b.shape[0] > 40 else b
            b2 = b[0:min(20, b.shape[0]), :, :] if b.shape[0] > 0 else b
            c1 = c[40:min(60, c.shape[0]), :, :] if c.shape[0] > 40 else c
            c2 = c[0:min(20, c.shape[0]), :, :] if c.shape[0] > 0 else c

        except:
            return None, None, None

        try:
            rects, processed_img = get_clz_rect_from_image(image_data, state)
            if not rects:
                return None, None, None

            v = get_cut_img(s, rects)
            if len(v) == 0:
                return None, None, None

            rs1 = get_flag_rect_emergency([a1, a2], v, 1)
            rs2 = get_flag_rect_emergency([b1, b2], v, 2)
            rs3 = get_flag_rect_emergency([c1, c2], v, 3)
            rs = rs1 + rs2 + rs3

            if len(rs) < 3:
                return None, None, None

            r = []
            used_types = set()

            for target_type in [1, 2, 3]:
                candidates = [x for x in rs if len(x) >= 3 and x[2] == target_type]
                if candidates:
                    best = max(candidates, key=lambda x: x[0])
                    r.append(best)
                    used_types.add(target_type)

            if len(r) >= 3:
                r = sorted(r[:3], key=lambda x: x[2])
                return r[0][1], r[1][1], r[2][1]
            else:
                return None, None, None

        except Exception as e:
            logger.error(f"Emergency: processing failed: {e}")
            return None, None, None

    except Exception as e:
        logger.error(f"Emergency: get_flags_rects_from_image failed: {e}")
        return None, None, None

# ============ OPTIMIZED IMAGE PROCESSING ============
def process_image_fast(image_data, state):
    """Optimized image processing with caching"""
    try:
        if not state or 'net' not in state:
            return [], None
            
        net = state['net']
        anchors = state.get('anchors', [])
        class_types = state.get('class_types', {})
        
        if not anchors or not class_types:
            return [], None
            
        # Fast decode
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        npimg = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if npimg is None:
            return [], None
            
        height, width = npimg.shape[:2]
        
        # Optimized resize
        npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
        npimg = cv2.resize(npimg, (416, 416), interpolation=cv2.INTER_LINEAR)
        
        # Faster preprocessing
        npimg_ = np.ascontiguousarray(np.transpose(npimg, (2, 1, 0)))
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(npimg_).unsqueeze(0).to(DEVICE)
            if USE_CUDA:
                input_tensor = input_tensor.half()
                
            y_pred = net(input_tensor)
            
            # Get predictions faster
            if USE_CUDA:
                y_pred = y_pred.cpu()
                
            v = parse_y_pred_fast(y_pred, anchors, class_types)
            
        ret = []
        rw, rh = width/416, height/416
        
        for item in v:
            if len(item) >= 3:
                rect, clz, con = item[0], item[1], item[2]
                rect[0] = int(rect[0] * rw)
                rect[2] = int(rect[2] * rw)
                rect[1] = int(rect[1] * rh)
                rect[3] = int(rect[3] * rh)
                ret.append([clz, rect])
                
        return ret, npimg
        
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return [], None

def parse_y_pred_fast(ypred, anchors, class_types, threshold=0.25):
    """Faster YOLO prediction parsing"""
    try:
        if not anchors or not class_types:
            return []
            
        ceillen = 5 + len(class_types)
        sigmoid = lambda x: 1/(1+math.exp(-x))
        infos = []
        
        # Process with numpy for speed
        ypred_np = ypred.numpy() if hasattr(ypred, 'numpy') else ypred
        
        for idx in range(min(len(anchors), 3)):
            try:
                tensor_idx = 4 + idx * ceillen
                if tensor_idx >= ypred_np.shape[3]:
                    continue
                    
                a = ypred_np[:, :, :, tensor_idx]
                
                # Vectorized operations
                for ii in range(a.shape[1]):
                    for jj in range(a.shape[2]):
                        val = a[0, ii, jj]
                        if val > threshold:
                            infos.append((ii, jj, idx, float(val)))
            except:
                continue
                
        if not infos:
            return []
            
        # Sort by confidence
        infos = sorted(infos, key=lambda i: -i[3])
        infos = infos[:20]  # Limit for speed
        
        results = []
        for info in infos:
            try:
                x, y, idx, con = info
                gp = idx * ceillen
                
                if (gp + 5 + len(class_types)) > ypred_np.shape[3]:
                    continue
                    
                # Get predictions
                pred_xy = ypred_np[0, x, y, gp:gp+2]
                pred_wh = ypred_np[0, x, y, gp+2:gp+4]
                pred_clz = ypred_np[0, x, y, gp+5:gp+5+len(class_types)]
                
                gap = 416/ypred_np.shape[1]
                
                cx, cy = float(pred_xy[0]), float(pred_xy[1])
                rx, ry = (cx + x) * gap, (cy + y) * gap
                rw = math.exp(float(pred_wh[0])) * anchors[idx][0]
                rh = math.exp(float(pred_wh[1])) * anchors[idx][1]
                
                xx = rx - rw/2
                _x = rx + rw/2
                yy = ry - rh/2
                _y = ry + rh/2
                
                # Get class
                clz = 'unknown'
                if pred_clz is not None and len(pred_clz) > 0:
                    max_val = max(pred_clz)
                    if max_val > 0.1:
                        max_idx = np.argmax(pred_clz)
                        for key, value in class_types.items():
                            if value == max_idx:
                                clz = key
                                break
                                
                results.append([[xx, yy, _x, _y], clz, con])
                
            except:
                continue
                
        return results
        
    except Exception as e:
        logger.error(f"Parse Y pred failed: {e}")
        return []

# ============ OPTIMIZED DUN163 CLASS ============
class Dun163Optimized:
    __slots__ = (
        'fp', 'resp_json2', 'domain', 'thread_id', '_current_image_data',
        '_current_rects', '_current_click_points', 'request_params',
        'ss', 'ctx', 'model_state', 'sift_detector'
    )
    
    def __init__(self, id_, *, referer, fp_h, ua, thread_id, domain=None):
        self.fp = None
        self.resp_json2 = None
        self.domain = domain if domain else random.choice(DUN163_DOMAINS)
        self.thread_id = thread_id
        self._current_image_data = None
        self._current_rects = None
        self._current_click_points = None
        
        self.request_params = {
            'id': id_,
            'referer': referer,
            'fp_h': fp_h,
            'ua': ua
        }
        self.ss = self.set_session()
        self.ctx = get_compiled_js('dun163.js')
        self.model_state = get_global_model()
        self.sift_detector = get_sift_detector()
        
    def set_session(self):
        session = requests.Session()
        session.headers.update({
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": self.request_params['referer'],
            "User-Agent": self.request_params['ua'],
        })
        # Use connection pooling
        session.mount('https://', requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=2,
            pool_block=False
        ))
        session.timeout = (CONNECTION_TIMEOUT, READ_TIMEOUT)
        return session

    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def get_jsonp(text):
        try:
            jsonp_str = re.search(r"\((.*)\)", text, re.S)
            if jsonp_str:
                return json.loads(jsonp_str.group(1))
            return {}
        except:
            return {}

    @staticmethod
    def random_jsonp_str():
        s = string.ascii_lowercase + string.digits
        return "__JSONP_" + ''.join(random.choices(s, k=7)) + '_'

    def request_getconf(self):
        try:
            url = self.domain + '/api/v2/getconf'
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "",
                "dt": "",
                "id": self.request_params['id'],
                "ipv6": "false",
                "runEnv": "10",
                "iv": "5",
                "loadVersion": "2.5.3",
                "lang": "en-US",
                "callback": self.random_jsonp_str() + '0'
            }
            response = self.ss.get(url, params=params, timeout=TOTAL_TIMEOUT)
            response.raise_for_status()
            resp_json = self.get_jsonp(response.text)
            return resp_json.get('data', {})
        except:
            return {}

    def request_get(self, dt, bid, ac_token, ir_token=None):
        try:
            url = self.domain + '/api/v3/get'
            
            # Get FP and CB in one call if possible
            self.fp = self.ctx.call('get_fp', self.request_params['fp_h'], self.request_params['ua'])
            cb = self.ctx.call('get_cb')
            
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "CN31",
                "dt": dt,
                "id": bid,
                "fp": self.fp,
                "https": "true",
                "version": "2.28.5",
                "dpr": "1",
                "dev": "1",
                "cb": cb,
                "ipv6": "false",
                "runEnv": "10",
                "lang": "en-US",
                "loadVersion": "2.5.3",
                "iv": "4",
                "width": "320",
                "audio": "false",
                "sizeType": "10",
                "smsVersion": "v3",
                "callback": self.random_jsonp_str() + '0'
            }
            
            if ir_token:
                params["irToken"] = ir_token
                
            resp_text = self.ss.get(url, params=params, timeout=TOTAL_TIMEOUT).text
            resp_json = self.get_jsonp(resp_text)
            return resp_json.get('data', {})
        except:
            return {}

    def request_check(self, dt, bid, *, token, captcha_type=7, click_data=None):
        try:
            url = self.domain + '/api/v3/check'
            
            if captcha_type == 7 and click_data:
                check_data = self.ctx.call('get_click_check_data', click_data, token)
            else:
                check_data = '{"d":"","m":"","p":"","ext":""}'
                
            cb = self.ctx.call('get_cb')
            
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "CN31",
                "dt": dt,
                "id": bid,
                "token": token,
                "data": check_data,
                "width": "320",
                "type": str(captcha_type),
                "version": "2.28.5",
                "cb": cb,
                "bf": "0",
                "runEnv": "10",
                "loadVersion": "2.5.3",
                "iv": "4",
                "callback": self.random_jsonp_str() + '1'
            }
            
            resp = self.ss.get(url, params=params, timeout=TOTAL_TIMEOUT)
            resp_json = self.get_jsonp(resp.text)
            return resp_json.get('data', {})
        except:
            return {}

    def handle_click_captcha_hybrid(self, bg_url, token, attempt_num=0):
        try:
            headers = {"User-Agent": self.request_params['ua']}
            resp = requests.get(bg_url, headers=headers, timeout=10)
            resp.raise_for_status()
            
            image_data = resp.content
            
            # Use cached model state
            state = self.model_state or get_global_model()
            if not state:
                return self.generate_emergency_clicks(), 0.0
                
            rects, _ = process_image_fast(image_data, state)
            
            self._current_image_data = image_data
            self._current_rects = rects
            
            rect1 = safe_list_access(rects, 0)
            rect2 = safe_list_access(rects, 1)
            rect3 = safe_list_access(rects, 2)
            
            if rect1 is not None and rect2 is not None and rect3 is not None:
                click_points = []
                for rect in [rect1, rect2, rect3]:
                    if rect and len(rect) >= 4:
                        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        offset_x = random.randint(-1, 1)
                        offset_y = random.randint(-1, 1)
                        final_x = max(5, min(center_x + offset_x, 315))
                        final_y = max(5, min(center_y + offset_y, 195))
                        click_points.append({"x": final_x, "y": final_y})
                        
                if len(click_points) >= 3:
                    self._current_click_points = click_points[:3]
                    return click_points[:3], 0.0
                    
            click_points = self.generate_emergency_clicks()
            self._current_click_points = click_points
            return click_points, 0.0
            
        except Exception as e:
            logger.debug(f"T-{self.thread_id} | Click handler error: {e}")
            return self.generate_emergency_clicks(), 0.0

    def generate_emergency_clicks(self):
        patterns = [
            [(80, 70), (160, 120), (240, 90)],
            [(70, 100), (160, 95), (250, 105)],
            [(160, 60), (110, 130), (210, 140)],
        ]
        selected = random.choice(patterns)
        return [{"x": min(305, max(15, x + random.randint(-5, 5))),
                 "y": min(185, max(15, y + random.randint(-5, 5)))} 
                for x, y in selected]

    def save_token_locally(self, validate_token):
        try:
            with file_lock:
                with open(TOKEN_OUTPUT_FILE, 'a') as f:
                    f.write(f"{validate_token}\n")
            return True
        except:
            return False

    def run(self, attempt_num=0):
        try:
            # Fast path - if model not loaded, skip
            if not self.model_state:
                return False
                
            get_conf_data = self.request_getconf()
            if not get_conf_data:
                return False
                
            dt = get_conf_data.get('dt')
            ac_data = get_conf_data.get('ac', {})
            ac_token = ac_data.get('token')
            bid = ac_data.get('bid')
            
            ir_data = get_conf_data.get('ir', {})
            ir_token = ir_data.get('token') if ir_data.get('enable') else None
            
            get_data = self.request_get(dt, bid, ac_token, ir_token)
            if not get_data:
                return False
                
            captcha_type = get_data.get('type', 7)
            token = get_data.get('token')
            
            if not token:
                return False
                
            if captcha_type == 7:
                bg_urls = get_data.get('bg', [])
                if not bg_urls:
                    return False
                    
                click_points, _ = self.handle_click_captcha_hybrid(bg_urls[0], token, attempt_num)
                resp_json = self.request_check(dt, bid, token=token, captcha_type=7, click_data=click_points)
            else:
                return False
                
            self.resp_json2 = resp_json
            
            if resp_json.get('result') == True:
                validate_raw = resp_json.get('validate', '')
                validate_decoded = ""
                
                if validate_raw and self.ctx:
                    try:
                        validate_decoded = self.ctx.call('do_onVerify', validate_raw, self.fp)
                    except:
                        return False
                        
                if validate_decoded and len(validate_decoded.strip()) > 10:
                    # Try batch first, then single
                    if not send_token_to_server(validate_decoded):
                        self.save_token_locally(validate_decoded)
                    return True
                return True
            else:
                return False
                
        except Exception as e:
            logger.debug(f"T-{self.thread_id} | Run error: {e}")
            return False

# ============ WORKER POOL ============
class WorkerPool:
    """Optimized worker pool with task sharing"""
    
    def __init__(self, num_workers=NUM_WORKERS):
        self.num_workers = num_workers
        self.workers = []
        self.running = False
        self.token_count = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def run_worker(self, worker_id, config):
        """Worker thread function optimized for speed"""
        session_start = time.time()
        success_count = 0
        failure_count = 0
        
        # Pre-create solver with warm start
        solver = Dun163Optimized(
            id_=config['ID_'],
            referer=config['REFERER'],
            fp_h=config['FP_H'],
            ua=config['UA'],
            thread_id=worker_id,
            domain=config['DOMAIN']
        )
        
        logger.info(f"T-{worker_id} | Started (UA: {config['UA'][:20]}...)")
        
        while self.running:
            try:
                # Adaptive timing
                if success_count > 10:
                    time.sleep(random.uniform(0.2, 0.5))
                elif failure_count > 3:
                    time.sleep(random.uniform(1.0, 2.0))
                    failure_count = 0
                else:
                    time.sleep(random.uniform(0.05, 0.15))
                
                success = solver.run()
                
                if success:
                    success_count += 1
                    with self.lock:
                        self.token_count += 1
                        if self.token_count % 10 == 0:
                            elapsed = time.time() - self.start_time
                            rate = self.token_count / elapsed if elapsed > 0 else 0
                            logger.info(f"📊 Rate: {rate:.2f} tokens/sec | Total: {self.token_count}")
                else:
                    failure_count += 1
                    
                    # Recreate solver on persistent failures
                    if failure_count > 5:
                        solver = Dun163Optimized(
                            id_=config['ID_'],
                            referer=config['REFERER'],
                            fp_h=config['FP_H'],
                            ua=UserAgent().random,
                            thread_id=worker_id,
                            domain=config['DOMAIN']
                        )
                        failure_count = 0
                        
            except Exception as e:
                logger.debug(f"T-{worker_id} | Worker error: {e}")
                failure_count += 1
                time.sleep(0.5)
                
    def start(self, config):
        """Start all workers"""
        self.running = True
        self.start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.num_workers):
                thread_config = config.copy()
                thread_config['UA'] = UserAgent().random
                thread_config['DOMAIN'] = DUN163_DOMAINS[i % len(DUN163_DOMAINS)]
                future = executor.submit(self.run_worker, i + 1, thread_config)
                futures.append(future)
                
            try:
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                self.running = False
                executor.shutdown(wait=False)
                raise

# ============ MAIN FUNCTION ============
def main():
    logger.info("=" * 60)
    logger.info("🚀 CN31 Solver - Optimized Edition")
    logger.info("=" * 60)
    
    # Initialize resources
    logger.info("Loading model...")
    model_state = initialize_global_model()
    if not model_state:
        logger.error("❌ Model not available - cannot continue")
        return
        
    logger.info("Loading JavaScript...")
    js_ctx = get_compiled_js('dun163.js')
    if not js_ctx:
        logger.error("❌ JavaScript not available - cannot continue")
        return
        
    logger.info("Initializing SIFT detector...")
    get_sift_detector()
    
    config = {
        'ID_': ID,
        'REFERER': REFERER,
        'FP_H': FP_H,
        'UA': UserAgent().random,
        'DOMAIN': DUN163_DOMAINS[0]
    }
    
    logger.info(f"🖥️  Device: {DEVICE}")
    logger.info(f"🔧 Workers: {NUM_WORKERS}")
    logger.info(f"🌐 Server: {TOKEN_SERVER_URL}")
    logger.info("-" * 60)
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    # Start worker pool
    pool = WorkerPool(num_workers=NUM_WORKERS)
    
    try:
        pool.start(config)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping workers...")
        pool.running = False
        
        # Print stats
        elapsed = time.time() - pool.start_time
        if elapsed > 0:
            logger.info(f"📊 Total tokens: {pool.token_count}")
            logger.info(f"📊 Rate: {pool.token_count / elapsed:.2f} tokens/sec")
            logger.info(f"📊 Time: {elapsed:.1f}s")
            
    except Exception as e:
        logger.error(f"❌ Main error: {e}")
        pool.running = False
        
    logger.info("👋 Done")

# ============ COMPATIBILITY WRAPPER ============
# Keep original class name for compatibility
Dun163 = Dun163Optimized

# Original worker function for compatibility
def worker_thread(thread_id, config):
    """Original worker thread function for backward compatibility"""
    pool = WorkerPool(num_workers=1)
    pool.running = True
    try:
        pool.run_worker(thread_id, config)
    finally:
        pool.running = False

if __name__ == '__main__':
    main()