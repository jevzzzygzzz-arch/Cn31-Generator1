# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.
import json
import os
import random
import re
import string
import time
import warnings
import math
import threading
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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

warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
warnings.filterwarnings("ignore", message=".*SIFT_create.*deprecated.*")

# Performance settings
DEBUG = False
TORCH_COMPILE = hasattr(torch, 'compile')
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
PIN_MEMORY = USE_CUDA

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TOKEN_SERVER_URL = os.environ.get('TOKEN_SERVER_URL', 'https://cn31-antrax-solver-production.up.railway.app')
TOKEN_SAVE_ENDPOINT = f"{TOKEN_SERVER_URL}/api/save-token"

# Optimized settings
TOKEN_BATCH_SIZE = 100
MAX_RETRIES = 2
POOL_SIZE = 200
POOL_CONNECTIONS = 200

# Global token batching
token_batch_queue = queue.Queue(maxsize=10000)
token_send_running = True
token_send_thread = None
token_stats = {"sent": 0, "failed": 0, "batched": 0}

# Session pool for requests
session_pool = queue.Queue(maxsize=POOL_SIZE)
session_lock = threading.Lock()

def create_session():
    """Create optimized session with connection pooling"""
    session = requests.Session()
    
    # Connection pooling
    adapter = HTTPAdapter(
        pool_connections=POOL_CONNECTIONS,
        pool_maxsize=POOL_CONNECTIONS,
        max_retries=Retry(
            total=1,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Optimized timeouts
    session.timeout = (2, 5)
    session.keep_alive = True
    
    return session

def get_session():
    """Get session from pool or create new"""
    try:
        return session_pool.get_nowait()
    except queue.Empty:
        return create_session()

def return_session(session):
    """Return session to pool"""
    try:
        session_pool.put_nowait(session)
    except queue.Full:
        session.close()

# Pre-populate session pool
for _ in range(min(POOL_SIZE // 2, 100)):
    session_pool.put(create_session())

def send_token_to_server_batch(tokens):
    """Send batch of tokens to server"""
    global token_stats
    success = 0
    failed = 0
    
    if not tokens:
        return 0, 0
    
    try:
        # Use session from pool
        session = get_session()
        try:
            payload = {"tokens": tokens}  # Batch send
            r = session.post(TOKEN_SAVE_ENDPOINT, json=payload, timeout=3)
            if r.status_code in [200, 201]:
                success = len(tokens)
            else:
                # Fallback to individual sends
                for token in tokens:
                    try:
                        r = session.post(TOKEN_SAVE_ENDPOINT, json={"token": token}, timeout=2)
                        if r.status_code in [200, 201]:
                            success += 1
                        else:
                            failed += 1
                    except:
                        failed += 1
        finally:
            return_session(session)
    except Exception as e:
        logger.error(f"Batch send error: {e}")
        failed = len(tokens)
    
    token_stats["sent"] += success
    token_stats["failed"] += failed
    return success, failed

def batch_token_sender():
    """Background thread to batch send tokens with high efficiency"""
    global token_send_running, token_stats
    batch = []
    last_send_time = time.time()
    
    while token_send_running:
        try:
            token = token_batch_queue.get(timeout=0.2)
            if token:
                batch.append(token)
                
            current_time = time.time()
            # Send batch when full or every 1 second
            if len(batch) >= TOKEN_BATCH_SIZE or (current_time - last_send_time > 1.0 and batch):
                if batch:
                    success, failed = send_token_to_server_batch(batch)
                    if success > 0:
                        logger.debug(f"Sent {success}/{len(batch)} tokens to server")
                    batch = []
                last_send_time = current_time
                
        except queue.Empty:
            # Send remaining after timeout
            if batch and time.time() - last_send_time > 2.0:
                success, failed = send_token_to_server_batch(batch)
                batch = []
                last_send_time = time.time()
        except Exception as e:
            logger.error(f"Batch sender error: {e}")
    
    # Final flush
    if batch:
        send_token_to_server_batch(batch)

# Start token sender thread
token_send_thread = threading.Thread(target=batch_token_sender, daemon=True)
token_send_thread.start()

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if TORCH_COMPILE:
        # Enable torch.compile for faster inference
        torch._dynamo.config.suppress_errors = True

def emergency_fallback():
    return [(80, 70), (160, 120), (240, 90)]

def safe_list_access(lst, index, default=None):
    try:
        if lst is None or not isinstance(lst, (list, tuple)):
            return default
        if not (0 <= index < len(lst)):
            return default
        return lst[index]
    except:
        return default

_model_state = None
_model_lock = threading.Lock()
_model_loaded = False

def initialize_global_model():
    global _model_state, _model_loaded
    if _model_loaded and _model_state is not None:
        return _model_state
    
    with _model_lock:
        if _model_loaded and _model_state is not None:
            return _model_state
            
        model_path = os.path.join(DIR_PATH, 'net.pkl')
        if not os.path.exists(model_path):
            logger.error("Model file net.pkl not found")
            return None
            
        try:
            # Load with memory mapping for faster loading
            state = torch.load(model_path, map_location=torch.device(DEVICE), weights_only=False)
            
            if 'net' in state:
                net = state['net']
                net = net.to(DEVICE)
                net.eval()
                
                if USE_CUDA:
                    net = net.half()
                    # Warm up CUDA
                    dummy = torch.randn(1, 3, 416, 416).half().cuda()
                    with torch.no_grad():
                        _ = net(dummy)
                    torch.cuda.synchronize()
                
                # Compile if available
                if TORCH_COMPILE and USE_CUDA:
                    try:
                        net = torch.compile(net, mode='reduce-overhead')
                        logger.info("Model compiled with torch.compile")
                    except:
                        pass
                
                state['net'] = net
                _model_state = state
                _model_loaded = True
                logger.success(f"Model loaded on {DEVICE} (compiled: {TORCH_COMPILE})")
                return _model_state
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return None

def get_global_model():
    global _model_state, _model_loaded
    if _model_loaded:
        return _model_state
    return initialize_global_model()

@lru_cache(maxsize=10)
def get_compiled_js_cached(file_name):
    try:
        js_path = os.path.join(DIR_PATH, file_name)
        with open(js_path, 'r', encoding='utf-8') as f:
            js_code = f.read()
        ctx = execjs.compile(js_code)
        return ctx
    except:
        return None

def get_compiled_js(file_name):
    return get_compiled_js_cached(file_name)

_sift_detector = None
_sift_lock = threading.Lock()

def get_sift_detector():
    global _sift_detector
    if _sift_detector is None:
        with _sift_lock:
            if _sift_detector is None:
                try:
                    _sift_detector = cv2.SIFT_create(nfeatures=20, contrastThreshold=0.1)
                except AttributeError:
                    try:
                        _sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures=20, contrastThreshold=0.1)
                    except:
                        _sift_detector = cv2.ORB_create(nfeatures=20)
    return _sift_detector

file_lock = threading.Lock()
TOKEN_OUTPUT_FILE = os.path.join(DIR_PATH, 'validated_tokens.txt')

REFERER = "https://mtacc.mobilelegends.com/"
ID = "fef5c67c39074e9d845f4bf579cc07af"
FP_H = "mtacc.mobilelegends.com"

DUN163_DOMAINS = [
    "https://c.dun.163.com",
    "https://c.dun.163yun.com"
]

# Pre-compute for performance
ROTATION_MATRIX_CACHE = {}

def rotate_about_center(src, angle, scale=1.):
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
    try:
        if not anchors or not class_types:
            return [] if islist else None
        ceillen = 5 + len(class_types)
        
        # Vectorized operations for speed
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
                    
                # Vectorized confidence extraction
                for ii, i in enumerate(a[0]):
                    for jj, j in enumerate(i):
                        if j > threshold * 0.5:  # Pre-filter
                            infos.append((ii, jj, idx, 1/(1+math.exp(-j))))
            except:
                continue
                
        if not infos:
            return [] if islist else None
            
        infos = sorted(infos, key=lambda i: -i[3])
        
        def get_xyxy_clz_con(info):
            try:
                gap = 416/ypred.shape[1]
                x, y, idx, con = info
                if idx >= len(anchors):
                    return None
                gp = idx * ceillen
                if (gp + 5 + len(class_types)) > ypred.shape[3]:
                    return None
                    
                # Batch operations
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
                
                cx, cy = float(pred_xy[0]), float(pred_xy[1])
                rx, ry = (cx + x)*gap, (cy + y)*gap
                rw, rh = math.exp(float(pred_wh[0]))*anchors[idx][0], math.exp(float(pred_wh[1]))*anchors[idx][1]
                
                # Class detection
                clz = 'unknown'
                if len(pred_clz) > 0:
                    max_idx = np.argmax(pred_clz)
                    for key, value in class_types.items():
                        if value == max_idx:
                            clz = key
                            break
                
                return [rx - rw/2, ry - rh/2, rx + rw/2, ry + rh/2], clz, con, None
            except:
                return None
                
        if islist:
            limited_infos = infos[:min(50, len(infos))]
            v = []
            for i in limited_infos:
                if i[3] > threshold:
                    result = get_xyxy_clz_con(i)
                    if result is not None:
                        v.append(result)
            return v
        else:
            if infos:
                return get_xyxy_clz_con(infos[0])
            return None
    except:
        return [] if islist else None

class Mini(nn.Module):
    class ConvBN(nn.Module):
        __slots__ = ['conv', 'bn', 'relu']
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
    except:
        return [], None

def get_cut_img(npimg, rects):
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

def get_flags_rects_from_image(image_data, state):
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
            
        def get_match_lens(i1, i2):
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
                
        def get_flag_rect(k12, cut_imgs, st):
            try:
                if len(k12) < 2:
                    return []
                k1, k2 = k12[0], k12[1]
                r = []
                for item in cut_imgs:
                    if len(item) >= 3:
                        clz, npimg, rect = item[0], item[1], item[2]
                        if clz == '1':
                            r1 = get_match_lens(k1, npimg)
                            r.append([r1, rect, st])
                        elif clz == '2':
                            r2 = get_match_lens(k2, npimg)
                            r.append([r2, rect, st])
                return sorted(r, key=lambda i: i[0]) if r else []
            except:
                return []
                
        try:
            rects, processed_img = get_clz_rect_from_image(image_data, state)
            if not rects:
                return None, None, None
                
            v = get_cut_img(s, rects)
            if len(v) == 0:
                return None, None, None
                
            rs1 = get_flag_rect([a1, a2], v, 1)
            rs2 = get_flag_rect([b1, b2], v, 2)
            rs3 = get_flag_rect([c1, c2], v, 3)
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
        except:
            return None, None, None
    except:
        return None, None, None

class Dun163:
    __slots__ = ['fp', 'resp_json2', 'domain', 'thread_id', '_current_image_data', 
                 '_current_rects', '_current_click_points', 'request_params', 'ss', 'ctx']
                 
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
        self.ss = None
        self.ctx = get_compiled_js('dun163.js')
        self._init_session()

    def _init_session(self):
        """Initialize session with connection pooling"""
        session = get_session()
        domain_host = self.domain.replace('https://', '').replace('http://', '')
        session.headers.update({
            "Accept": "*/*",
            "Accept-Language": "*",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Referer": self.request_params['referer'],
            "User-Agent": self.request_params['ua'],
            "Host": domain_host,
        })
        self.ss = session

    @staticmethod
    @lru_cache(maxsize=2000)
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
            response = self.ss.get(url, params=params, timeout=5)
            response.raise_for_status()
            return self.get_jsonp(response.text).get('data', {})
        except:
            return {}

    def request_get(self, dt, bid, ac_token, ir_token=None):
        try:
            url = self.domain + '/api/v3/get'
            
            if not self.ctx:
                self.ctx = get_compiled_js('dun163.js')
                
            fp = self.ctx.call('get_fp', self.request_params['fp_h'], self.request_params['ua'])
            cb = self.ctx.call('get_cb')
            self.fp = fp
            
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "CN31",
                "dt": dt,
                "id": bid,
                "fp": fp,
                "https": "true",
                "type": "",
                "version": "2.28.5",
                "dpr": "1",
                "dev": "1",
                "cb": cb,
                "ipv6": "false",
                "runEnv": "10",
                "group": "",
                "scene": "",
                "lang": "en-US",
                "sdkVersion": "",
                "loadVersion": "2.5.3",
                "iv": "4",
                "user": "",
                "width": "320",
                "audio": "false",
                "sizeType": "10",
                "smsVersion": "v3",
                "token": "",
                "callback": self.random_jsonp_str() + '0'
            }
            if ir_token:
                params["irToken"] = ir_token
                
            resp_text = self.ss.get(url, params=params, timeout=5).text
            return self.get_jsonp(resp_text).get('data', {})
        except:
            return {}

    def request_check(self, dt, bid, *, token, captcha_type=7, click_data=None):
        try:
            url = self.domain + '/api/v3/check'
            
            if captcha_type == 7 and click_data and self.ctx:
                check_data = self.ctx.call('get_click_check_data', click_data, token)
            else:
                check_data = '{"d":"","m":"","p":"","ext":""}'
                
            cb = self.ctx.call('get_cb') if self.ctx else '1'
            
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
                "user": "",
                "extraData": "",
                "bf": "0",
                "runEnv": "10",
                "sdkVersion": "",
                "loadVersion": "2.5.3",
                "iv": "4",
                "callback": self.random_jsonp_str() + '1'
            }
            
            resp = self.ss.get(url, params=params, timeout=5)
            return self.get_jsonp(resp.text).get('data', {})
        except:
            return {}

    def handle_click_captcha(self, bg_url, token, attempt_num=0):
        try:
            headers = {"User-Agent": self.request_params['ua']}
            
            # Use session from pool for image download
            session = get_session()
            try:
                resp = session.get(bg_url, headers=headers, timeout=5)
                resp.raise_for_status()
                image_data = resp.content
            finally:
                return_session(session)
                
            state = get_global_model()
            if not state:
                return self.generate_emergency_clicks(), 0.0
                
            start_time = time.time()
            rects = get_flags_rects_from_image(image_data, state)
            img_time = time.time() - start_time
            
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
                    return click_points[:3], img_time
                    
            click_points = self.generate_emergency_clicks()
            self._current_click_points = click_points
            return click_points, img_time
        except:
            return self.generate_emergency_clicks(), 0.0

    def generate_emergency_clicks(self):
        try:
            patterns = [
                [(80, 70), (160, 120), (240, 90)],
                [(70, 100), (160, 95), (250, 105)],
                [(160, 60), (110, 130), (210, 140)],
                [(100, 80), (180, 110), (260, 85)],
                [(90, 130), (150, 75), (230, 115)],
            ]
            selected_pattern = random.choice(patterns)
            click_points = []
            for x, y in selected_pattern:
                offset_x = random.randint(-3, 3)
                offset_y = random.randint(-3, 3)
                final_x = max(15, min(x + offset_x, 305))
                final_y = max(15, min(y + offset_y, 185))
                click_points.append({"x": final_x, "y": final_y})
            return click_points
        except:
            return [{"x": 80, "y": 70}, {"x": 160, "y": 120}, {"x": 240, "y": 90}]

    def save_token_locally(self, validate_token):
        try:
            line = f"{validate_token}\n"
            with file_lock:
                with open(TOKEN_OUTPUT_FILE, 'a') as f:
                    f.write(line)
            return True
        except:
            return False

    def run(self, attempt_num=0):
        try:
            # Refresh session periodically
            if attempt_num > 0 and attempt_num % 15 == 0:
                if self.ss:
                    return_session(self.ss)
                self._init_session()
                self.ctx = get_compiled_js('dun163.js')
                
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
                    
                click_points, img_time = self.handle_click_captcha(bg_urls[0], token, attempt_num)
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
                    # Queue token for batch sending
                    try:
                        token_batch_queue.put_nowait(validate_decoded)
                    except queue.Full:
                        # Queue full, save locally
                        self.save_token_locally(validate_decoded)
                    return True
                else:
                    return True
            else:
                return False
        except Exception as e:
            logger.error(f'T-{self.thread_id} | Error: {str(e)[:80]}')
            return False

def worker_thread(thread_id, config):
    """Optimized worker with aggressive retry strategy"""
    d = None
    consecutive_failures = 0
    success_count = 0
    fail_count = 0
    
    while True:
        try:
            if d is None or consecutive_failures > 10:
                d = Dun163(
                    id_=config['ID_'],
                    referer=config['REFERER'],
                    fp_h=config['FP_H'],
                    ua=config['UA'],
                    thread_id=thread_id,
                    domain=config['DOMAIN']
                )
                consecutive_failures = 0
                
            # Run without delays
            success = d.run()
            
            if success:
                success_count += 1
                consecutive_failures = 0
                if success_count % 50 == 0:
                    logger.info(f"T-{thread_id} | Success: {success_count}, Failures: {fail_count}")
            else:
                consecutive_failures += 1
                fail_count += 1
                if consecutive_failures > 10:
                    # Recreate solver
                    if d and d.ss:
                        return_session(d.ss)
                    d = None
                    
        except Exception as e:
            logger.error(f"T-{thread_id} | Worker error: {e}")
            d = None
            time.sleep(0.1)

def main():
    logger.info("Starting CN31 Solver Ultra...")
    
    # Initialize model
    model_state = initialize_global_model()
    if not model_state:
        logger.error("Model not available - cannot continue")
        return
        
    # Pre-warm all components
    get_compiled_js('dun163.js')
    get_sift_detector()
    logger.success("All resources loaded")
    
    config = {
        'ID_': ID,
        'REFERER': REFERER,
        'FP_H': FP_H,
        'UA': UserAgent().random,
        'DOMAIN': random.choice(DUN163_DOMAINS)
    }
    
    NUM_THREADS = 100
    
    logger.info(f"Starting {NUM_THREADS} worker threads")
    logger.info(f"ID: {ID}")
    logger.info(f"Server: {TOKEN_SERVER_URL}")
    logger.info("-" * 50)
    
    # Start workers with staggered startup
    threads = []
    for i in range(NUM_THREADS):
        thread_config = config.copy()
        thread_config['UA'] = UserAgent().random
        thread_config['DOMAIN'] = DUN163_DOMAINS[i % len(DUN163_DOMAINS)]
        t = threading.Thread(target=worker_thread, args=(i+1, thread_config), daemon=True)
        t.start()
        threads.append(t)
        if i % 10 == 0:
            time.sleep(0.01)
    
    # Monitor thread status
    try:
        while True:
            time.sleep(30)
            logger.info(f"Active: {threading.active_count()}, Queue: {token_batch_queue.qsize()}, Sent: {token_stats['sent']}")
    except KeyboardInterrupt:
        logger.warning("Stopping...")
        global token_send_running
        token_send_running = False
        sys.exit(0)

if __name__ == '__main__':
    main()