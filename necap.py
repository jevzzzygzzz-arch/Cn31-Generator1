# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.

import os
import sys
import time
import json
import threading
import queue
from datetime import datetime
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
import random

app = Flask(__name__)
CORS(app)

sys.path.insert(0, '/app')

# Configuration
DEFAULT_WORKERS = int(os.environ.get('CN31_WORKERS', 50))
MAX_WORKERS = int(os.environ.get('CN31_MAX_WORKERS', 100))
TOKEN_CACHE_SIZE = 1000

try:
    from yidun_proxyless import *
    import yidun_proxyless as solver
    SOLVER_AVAILABLE = True
    print("✅ CN31 Solver loaded successfully")

    if os.path.exists('/app/dun163.js'):
        print("✅ dun163.js found")
    else:
        print("❌ dun163.js NOT found")
        SOLVER_AVAILABLE = False

    if os.path.exists('/app/net.pkl'):
        size = os.path.getsize('/app/net.pkl')
        print(f"✅ net.pkl found ({size} bytes)")
    else:
        print("❌ net.pkl NOT found")
        SOLVER_AVAILABLE = False

except ImportError as e:
    SOLVER_AVAILABLE = False
    print(f"❌ CN31 Solver not available: {e}")
    import traceback
    traceback.print_exc()

# Global state
solver_running = False
solver_thread = None
solver_executor = None
generation_stats = {
    "status": "idle",
    "tokens_generated": 0,
    "tokens_per_second": 0,
    "start_time": None,
    "threads": 0,
    "active_threads": 0,
    "queue_size": 0
}

# Token management
tokens_cache = []
token_lock = threading.Lock()
token_queue = queue.Queue(maxsize=TOKEN_CACHE_SIZE)
TOKEN_FILE = "/app/validated_tokens.txt"

# Statistics
stats_lock = threading.Lock()
total_tokens = 0
start_time = None
tokens_per_second = 0
token_counter_interval = 0
last_stats_update = time.time()

def read_tokens_from_file():
    """Read tokens from the validated tokens file"""
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r') as f:
                lines = f.readlines()
                return [line.strip() for line in lines if line.strip()]
        return []
    except Exception as e:
        print(f"Error reading tokens: {e}")
        return []

def get_new_tokens():
    """Get new tokens from file and add to queue"""
    global tokens_cache, total_tokens

    try:
        current_tokens = set(tokens_cache)
        file_tokens = set(read_tokens_from_file())
        new_tokens = file_tokens - current_tokens

        if new_tokens:
            with token_lock:
                for token in new_tokens:
                    try:
                        token_queue.put_nowait(token)
                        tokens_cache.append(token)
                    except queue.Full:
                        break
                total_tokens = len(tokens_cache)
                generation_stats["tokens_generated"] = total_tokens
            print(f"✅ Added {len(new_tokens)} new tokens. Total: {total_tokens}")

        return list(new_tokens)
    except Exception as e:
        print(f"Error getting new tokens: {e}")
        return []

def update_stats():
    """Update token generation statistics"""
    global total_tokens, start_time, tokens_per_second, last_stats_update, token_counter_interval

    with stats_lock:
        now = time.time()
        if start_time is None:
            start_time = now
            return

        elapsed = now - start_time
        if elapsed > 0:
            tokens_per_second = total_tokens / elapsed
            generation_stats["tokens_per_second"] = round(tokens_per_second, 2)
            generation_stats["tokens_generated"] = total_tokens
            generation_stats["uptime"] = elapsed

def run_solver_worker(threads=3):
    """Run the solver in the background"""
    global solver_running, generation_stats, start_time

    print(f"🚀 Starting CN31 solver with {threads} threads...")
    start_time = time.time()
    generation_stats["status"] = "running"
    generation_stats["start_time"] = datetime.now().isoformat()
    generation_stats["threads"] = threads

    # Set the number of threads in the solver module
    solver.MAX_WORKERS = min(threads, MAX_WORKERS)
    solver.NUM_THREADS = solver.MAX_WORKERS

    try:
        # Start batch processor
        try:
            solver.start_batch_processor()
        except AttributeError:
            pass
            
        # Run the solver
        solver.main()
    except KeyboardInterrupt:
        print("⏹️ Solver stopped by user")
    except Exception as e:
        print(f"❌ Solver error: {e}")
        generation_stats["status"] = "error"
        generation_stats["error"] = str(e)
    finally:
        solver_running = False
        generation_stats["status"] = "stopped"

@app.route('/')
def status():
    """Get current status"""
    get_new_tokens()
    update_stats()

    return jsonify({
        "status": generation_stats["status"],
        "tokens_generated": generation_stats["tokens_generated"],
        "tokens_in_queue": token_queue.qsize(),
        "tokens_per_second": generation_stats["tokens_per_second"],
        "threads": generation_stats["threads"],
        "start_time": generation_stats.get("start_time"),
        "uptime": generation_stats.get("uptime", 0),
        "solver_available": SOLVER_AVAILABLE,
        "files_ready": all([
            os.path.exists('/app/yidun_proxyless.py'),
            os.path.exists('/app/dun163.js'),
            os.path.exists('/app/net.pkl')
        ])
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    get_new_tokens()
    return jsonify({
        "ok": True,
        "solver_available": SOLVER_AVAILABLE,
        "status": generation_stats["status"],
        "tokens_available": token_queue.qsize(),
        "files": {
            "yidun_proxyless.py": os.path.exists('/app/yidun_proxyless.py'),
            "dun163.js": os.path.exists('/app/dun163.js'),
            "net.pkl": os.path.exists('/app/net.pkl')
        }
    })

@app.route('/debug/files')
def debug_files():
    """Debug file status"""
    import os
    files = {
        'yidun_proxyless.py': os.path.exists('/app/yidun_proxyless.py'),
        'dun163.js': os.path.exists('/app/dun163.js'),
        'net.pkl': os.path.exists('/app/net.pkl'),
        'validated_tokens.txt': os.path.exists('/app/validated_tokens.txt'),
    }

    sizes = {}
    for f in files:
        if files[f]:
            try:
                sizes[f] = os.path.getsize(f'/app/{f}')
            except:
                sizes[f] = 'error'

    return jsonify({
        'files': files,
        'sizes': sizes,
        'cwd': os.getcwd(),
        'all_files': os.listdir('/app') if os.path.exists('/app') else []
    })

@app.route('/debug/model')
def debug_model():
    """Debug model loading"""
    try:
        import torch
        import os

        if not os.path.exists('/app/net.pkl'):
            return jsonify({
                'error': 'net.pkl not found',
                'files': os.listdir('/app') if os.path.exists('/app') else []
            })

        model_path = '/app/net.pkl'
        model = torch.load(model_path, map_location='cpu')

        return jsonify({
            'model_loaded': True,
            'model_keys': list(model.keys()) if hasattr(model, 'keys') else 'N/A',
            'model_path': model_path,
            'model_size': os.path.getsize(model_path)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'model_loaded': False
        })

@app.route('/start', methods=['POST'])
def start_solver():
    """Start the solver with specified number of threads"""
    global solver_running, solver_thread, generation_stats

    if solver_running:
        return jsonify({"error": "Solver already running"}), 400

    if not SOLVER_AVAILABLE:
        return jsonify({"error": "CN31 Solver not available"}), 500

    required_files = ['yidun_proxyless.py', 'dun163.js', 'net.pkl']
    missing = [f for f in required_files if not os.path.exists(f'/app/{f}')]

    if missing:
        return jsonify({
            "error": f"Missing files: {missing}",
            "files": os.listdir('/app') if os.path.exists('/app') else []
        }), 500

    data = request.json or {}
    threads = min(data.get("threads", DEFAULT_WORKERS), MAX_WORKERS)

    try:
        model = initialize_global_model()
        if model is None:
            return jsonify({"error": "Failed to load model (check /debug/model)"}), 500
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500

    solver_running = True
    generation_stats["status"] = "starting"
    generation_stats["threads"] = threads

    solver_thread = threading.Thread(
        target=run_solver_worker,
        args=(threads,),
        daemon=False
    )
    solver_thread.start()

    return jsonify({
        "message": "CN31 Solver started",
        "threads": threads,
        "status": "running"
    })

@app.route('/stop', methods=['POST'])
def stop_solver():
    """Stop the solver"""
    global solver_running

    solver_running = False
    generation_stats["status"] = "stopping"

    return jsonify({
        "message": "Stop signal sent",
        "tokens_generated": generation_stats["tokens_generated"]
    })

@app.route('/api/get-token', methods=['GET'])
def get_token():
    """Get a single token from the queue"""
    get_new_tokens()

    try:
        token = token_queue.get(timeout=0.5)
        return jsonify({
            "token": token,
            "remaining": token_queue.qsize()
        })
    except queue.Empty:
        return jsonify({"error": "No tokens available"}), 404

@app.route('/api/tokens', methods=['GET'])
def get_tokens():
    """Get multiple tokens from the queue"""
    global tokens_cache

    n = request.args.get('n', 5, type=int)
    n = min(n, 50)

    get_new_tokens()

    tokens = []
    try:
        for _ in range(n):
            token = token_queue.get(timeout=0.1)
            tokens.append(token)
    except queue.Empty:
        pass

    if not tokens:
        return jsonify({"error": "No tokens available"}), 404

    return jsonify({
        "tokens": tokens,
        "count": len(tokens),
        "remaining": token_queue.qsize()
    })

@app.route('/api/stream-tokens')
def stream_tokens():
    """Stream tokens as they become available (Server-Sent Events)"""
    def generate():
        get_new_tokens()
        while True:
            try:
                token = token_queue.get(timeout=1.0)
                yield f"data: {json.dumps({'token': token, 'remaining': token_queue.qsize()})}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True, 'remaining': token_queue.qsize()})}\n\n"
                continue

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detailed statistics"""
    get_new_tokens()
    update_stats()

    return jsonify({
        "tokens_generated": generation_stats["tokens_generated"],
        "tokens_available": token_queue.qsize(),
        "tokens_per_second": generation_stats["tokens_per_second"],
        "status": generation_stats["status"],
        "threads": generation_stats["threads"],
        "start_time": generation_stats.get("start_time"),
        "uptime": generation_stats.get("uptime", 0),
        "queue_capacity": TOKEN_CACHE_SIZE,
        "solver_available": SOLVER_AVAILABLE
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update solver configuration"""
    global DEFAULT_WORKERS, MAX_WORKERS, TOKEN_CACHE_SIZE
    
    data = request.json or {}
    
    if 'workers' in data:
        new_workers = min(data['workers'], 100)
        DEFAULT_WORKERS = new_workers
        os.environ['CN31_WORKERS'] = str(new_workers)
    
    if 'queue_size' in data:
        new_queue = min(data['queue_size'], 5000)
        TOKEN_CACHE_SIZE = new_queue
        
    return jsonify({
        "workers": DEFAULT_WORKERS,
        "max_workers": MAX_WORKERS,
        "queue_size": TOKEN_CACHE_SIZE
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 6000))

    print(f"""
🔐 CN31 Solver - Optimized Railway Edition
─────────────────────────────────────────
Port          : {port}
Workers       : {DEFAULT_WORKERS} (max: {MAX_WORKERS})
Queue Size    : {TOKEN_CACHE_SIZE}
Solver        : {'✅ Available' if SOLVER_AVAILABLE else '❌ Not Available'}
Files:
  - yidun_proxyless.py: {'✅' if os.path.exists('/app/yidun_proxyless.py') else '❌'}
  - dun163.js: {'✅' if os.path.exists('/app/dun163.js') else '❌'}
  - net.pkl: {'✅' if os.path.exists('/app/net.pkl') else '❌'}
─────────────────────────────────────────
API Endpoints:
  GET  /                 - Status
  GET  /health           - Health check
  POST /start            - Start solver
  POST /stop             - Stop solver
  GET  /api/get-token    - Get one token
  GET  /api/tokens?n=5   - Get multiple tokens
  GET  /api/stream-tokens - Stream tokens (SSE)
  GET  /api/stats        - Statistics
  POST /api/config       - Update config
""")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)