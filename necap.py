# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.

import os
import sys
import time
import json
import threading
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

sys.path.insert(0, '/app')

try:
    from yidun_proxyless import *
    import yidun_proxyless as solver
    SOLVER_AVAILABLE = True
    print(" CN31 Solver loaded successfully")

    if os.path.exists('/app/dun163.js'):
        print(" dun163.js found")
    else:
        print(" dun163.js NOT found")
        SOLVER_AVAILABLE = False

    if os.path.exists('/app/net.pkl'):
        size = os.path.getsize('/app/net.pkl')
        print(f" net.pkl found ({size} bytes)")
    else:
        print(" net.pkl NOT found")
        SOLVER_AVAILABLE = False

except ImportError as e:
    SOLVER_AVAILABLE = False
    print(f" CN31 Solver not available: {e}")
    import traceback
    traceback.print_exc()

solver_running = False
solver_thread = None
generation_stats = {
    "status": "idle",
    "tokens_generated": 0,
    "tokens_per_second": 0,
    "start_time": None,
    "threads": 0,
    "success_count": 0,
    "failure_count": 0
}

tokens_cache = []
token_lock = threading.Lock()
TOKEN_FILE = "/app/validated_tokens.txt"

_start_time = None
_token_count = 0
_metrics_lock = threading.Lock()

def read_tokens_from_file():
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
    global tokens_cache
    try:
        current_tokens = set(tokens_cache)
        file_tokens = set(read_tokens_from_file())
        new_tokens = file_tokens - current_tokens

        if new_tokens:
            with token_lock:
                tokens_cache.extend(list(new_tokens))
                generation_stats["tokens_generated"] = len(tokens_cache)
            print(f" Added {len(new_tokens)} new tokens. Total: {len(tokens_cache)}")
        return list(new_tokens)
    except Exception as e:
        print(f"Error getting new tokens: {e}")
        return []

def run_solver_worker(threads=80):
    global solver_running, generation_stats, _start_time

    print(f" Starting CN31 solver with {threads} threads ...")
    generation_stats["status"] = "running"
    generation_stats["start_time"] = datetime.now().isoformat()
    generation_stats["threads"] = threads
    _start_time = time.time()

    try:
        solver.run_workers(threads)
    except KeyboardInterrupt:
        print(" Solver stopped by user")
    except Exception as e:
        print(f" Solver error: {e}")
        generation_stats["status"] = "error"
        generation_stats["error"] = str(e)
    finally:
        solver_running = False
        generation_stats["status"] = "stopped"

@app.route('/')
def status():
    get_new_tokens()
    
    elapsed = time.time() - _start_time if _start_time else 0
    tps = generation_stats["tokens_generated"] / elapsed if elapsed > 0 else 0
    
    return jsonify({
        "status": generation_stats["status"],
        "tokens_generated": generation_stats["tokens_generated"],
        "tokens_in_queue": len(tokens_cache),
        "tokens_per_second": round(tps, 2),
        "threads": generation_stats["threads"],
        "start_time": generation_stats.get("start_time"),
        "uptime_seconds": round(elapsed, 1),
        "solver_available": SOLVER_AVAILABLE,
        "files_ready": all([
            os.path.exists('/app/yidun_proxyless.py'),
            os.path.exists('/app/dun163.js'),
            os.path.exists('/app/net.pkl')
        ])
    })

@app.route('/health')
def health():
    return jsonify({
        "ok": True,
        "solver_available": SOLVER_AVAILABLE,
        "status": generation_stats["status"],
        "tokens_available": len(tokens_cache),
        "files": {
            "yidun_proxyless.py": os.path.exists('/app/yidun_proxyless.py'),
            "dun163.js": os.path.exists('/app/dun163.js'),
            "net.pkl": os.path.exists('/app/net.pkl')
        }
    })

@app.route('/debug/files')
def debug_files():
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
    threads = min(data.get("threads", 80), 100)

    try:
        model = initialize_global_model()
        if model is None:
            return jsonify({"error": "Failed to load model (check /debug/model)"}), 500
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500

    solver_running = True
    generation_stats["status"] = "starting"
    generation_stats["threads"] = threads
    generation_stats["tokens_generated"] = 0

    solver_thread = threading.Thread(
        target=run_solver_worker,
        args=(threads,),
        daemon=False
    )
    solver_thread.start()

    return jsonify({
        "message": f" CN31 Solver started with {threads} workers",
        "threads": threads,
        "status": "running"
    })

@app.route('/stop', methods=['POST'])
def stop_solver():
    global solver_running
    
    solver._stop_flag = True
    solver_running = False
    generation_stats["status"] = "stopping"

    return jsonify({
        "message": "Stop signal sent",
        "tokens_generated": generation_stats["tokens_generated"]
    })

@app.route('/api/get-token', methods=['GET'])
def get_token():
    global tokens_cache
    get_new_tokens()
    with token_lock:
        if tokens_cache:
            token = tokens_cache.pop(0)
            return jsonify({
                "token": token,
                "remaining": len(tokens_cache)
            })
    return jsonify({"error": "No tokens available"}), 404

@app.route('/api/tokens', methods=['GET'])
def get_tokens():
    global tokens_cache
    n = request.args.get('n', 5, type=int)
    n = min(n, 50)
    get_new_tokens()
    with token_lock:
        count = min(n, len(tokens_cache))
        result = tokens_cache[:count]
        tokens_cache = tokens_cache[count:]
        return jsonify({
            "tokens": result,
            "count": len(result),
            "remaining": len(tokens_cache)
        })

@app.route('/api/stats')
def stats():
    """Get detailed performance statistics"""
    elapsed = time.time() - _start_time if _start_time else 0
    tps = generation_stats["tokens_generated"] / elapsed if elapsed > 0 else 0
    
    return jsonify({
        "tokens_generated": generation_stats["tokens_generated"],
        "tokens_per_second": round(tps, 2),
        "uptime_seconds": round(elapsed, 1),
        "queue_size": len(tokens_cache),
        "threads": generation_stats["threads"],
        "status": generation_stats["status"]
    })

@app.route('/api/ping')
def ping():
    """Simple ping endpoint for health checks"""
    return jsonify({"pong": True, "time": time.time()})

@app.route('/api/reset', methods=['POST'])
def reset_solver():
    """Reset the solver and clear tokens"""
    global solver_running, tokens_cache, generation_stats
    
    solver._stop_flag = True
    solver_running = False
    
    with token_lock:
        tokens_cache = []
        generation_stats["tokens_generated"] = 0
        generation_stats["status"] = "reset"
    
    return jsonify({
        "message": "Solver reset successfully",
        "tokens_cleared": True
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 6000))
    
    print(f"""
 CN31 Solver - (80 Workers)
═══════════════════════════════════════════════════════════
Port       : {port}
Solver     : {'✅ Available' if SOLVER_AVAILABLE else '❌ Not Available'}
Workers    : 80 (MAXIMUM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Files:
  ├─ yidun_proxyless.py: {'✅' if os.path.exists('/app/yidun_proxyless.py') else '❌'}
  ├─ dun163.js: {'✅' if os.path.exists('/app/dun163.js') else '❌'}
  ├─ net.pkl: {'✅' if os.path.exists('/app/net.pkl') else '❌'}
  └─ validated_tokens.txt: {'✅' if os.path.exists('/app/validated_tokens.txt') else ' (will be created)'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 To start: POST /start with {{"threads": 80}}
 Monitor: GET /api/stats
 Get tokens: GET /api/get-token
 Reset: POST /api/reset
═══════════════════════════════════════════════════════════
""")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)