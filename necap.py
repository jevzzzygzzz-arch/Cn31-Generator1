# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.

import os
import sys
import time
import json
import threading
import random
from datetime import datetime
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import gc

app = Flask(__name__)
CORS(app)

sys.path.insert(0, '/app')

# Global flags
SOLVER_AVAILABLE = False
solver_running = False
solver_threads = []
token_send_running = True

# Stats
generation_stats = {
    "status": "idle",
    "tokens_generated": 0,
    "tokens_sent": 0,
    "start_time": None,
    "threads": 0,
    "tokens_per_second": 0,
    "uptime_seconds": 0
}

tokens_cache = []
token_lock = threading.Lock()
TOKEN_FILE = "/app/validated_tokens.txt"
stats_lock = threading.Lock()

try:
    from yidun_proxyless import *
    import yidun_proxyless as solver
    SOLVER_AVAILABLE = True
    print("✅ CN31 Solver loaded successfully")
    
    # Check required files
    required_files = ['dun163.js', 'net.pkl']
    missing = []
    for f in required_files:
        if os.path.exists(f'/app/{f}'):
            print(f"✅ {f} found")
        else:
            print(f"❌ {f} NOT found")
            missing.append(f)
            
    if missing:
        SOLVER_AVAILABLE = False
        print(f"❌ Missing files: {missing}")

except ImportError as e:
    SOLVER_AVAILABLE = False
    print(f"❌ CN31 Solver not available: {e}")
    import traceback
    traceback.print_exc()

def read_tokens_from_file():
    """Read tokens from file with buffering"""
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r', buffering=8192) as f:
                lines = f.readlines()
                # Filter and strip in one pass
                return [line.strip() for line in lines if line.strip()]
        return []
    except Exception as e:
        print(f"Error reading tokens: {e}")
        return []

def get_new_tokens():
    """Get new tokens efficiently"""
    global tokens_cache, generation_stats
    
    try:
        file_tokens = read_tokens_from_file()
        if not file_tokens:
            return []
            
        with token_lock:
            # Use set difference for speed
            current_set = set(tokens_cache)
            new_tokens = [t for t in file_tokens if t not in current_set]
            
            if new_tokens:
                tokens_cache.extend(new_tokens)
                generation_stats["tokens_generated"] = len(tokens_cache)
                print(f"✅ Added {len(new_tokens)} new tokens. Total: {len(tokens_cache)}")
                
        return new_tokens
    except Exception as e:
        print(f"Error getting new tokens: {e}")
        return []

def run_solver_worker(threads=100):
    """Run solver with specified thread count"""
    global solver_running, generation_stats
    
    print(f"🚀 Starting CN31 solver with {threads} threads...")
    generation_stats["status"] = "running"
    generation_stats["start_time"] = datetime.now().isoformat()
    generation_stats["threads"] = threads
    generation_stats["tokens_per_second"] = 0
    
    try:
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
    """Main status endpoint"""
    get_new_tokens()
    
    # Calculate tokens per second
    with stats_lock:
        if generation_stats.get("start_time"):
            start = datetime.fromisoformat(generation_stats["start_time"])
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed > 0:
                generation_stats["tokens_per_second"] = generation_stats["tokens_generated"] / elapsed
                generation_stats["uptime_seconds"] = int(elapsed)
    
    return jsonify({
        "status": generation_stats["status"],
        "tokens_generated": generation_stats["tokens_generated"],
        "tokens_in_queue": len(tokens_cache),
        "threads": generation_stats["threads"],
        "start_time": generation_stats.get("start_time"),
        "tokens_per_second": round(generation_stats.get("tokens_per_second", 0), 2),
        "uptime_seconds": generation_stats.get("uptime_seconds", 0),
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
    return jsonify({
        "ok": True,
        "solver_available": SOLVER_AVAILABLE,
        "status": generation_stats["status"],
        "tokens_available": len(tokens_cache),
        "threads_active": threading.active_count(),
        "files": {
            "yidun_proxyless.py": os.path.exists('/app/yidun_proxyless.py'),
            "dun163.js": os.path.exists('/app/dun163.js'),
            "net.pkl": os.path.exists('/app/net.pkl')
        }
    })

@app.route('/debug/files')
def debug_files():
    """Debug file status"""
    files = {
        'yidun_proxyless.py': os.path.exists('/app/yidun_proxyless.py'),
        'dun163.js': os.path.exists('/app/dun163.js'),
        'net.pkl': os.path.exists('/app/net.pkl'),
        'validated_tokens.txt': os.path.exists('/app/validated_tokens.txt'),
    }
    
    sizes = {}
    for f, exists in files.items():
        if exists:
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
            'model_size': os.path.getsize(model_path),
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'model_loaded': False
        })

@app.route('/start', methods=['POST'])
def start_solver():
    """Start the solver"""
    global solver_running, solver_threads, generation_stats
    
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
    threads = min(data.get("threads", 100), 200)
    
    # Validate model
    try:
        model = initialize_global_model()
        if model is None:
            return jsonify({"error": "Failed to load model (check /debug/model)"}), 500
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500
    
    # Reset stats
    generation_stats["tokens_generated"] = 0
    generation_stats["tokens_sent"] = 0
    generation_stats["tokens_per_second"] = 0
    
    solver_running = True
    generation_stats["status"] = "starting"
    generation_stats["threads"] = threads
    
    # Start solver
    solver_thread = threading.Thread(
        target=run_solver_worker,
        args=(threads,),
        daemon=False
    )
    solver_thread.start()
    solver_threads.append(solver_thread)
    
    return jsonify({
        "message": "CN31 Solver started",
        "threads": threads,
        "status": "running",
        "max_threads": 200
    })

@app.route('/stop', methods=['POST'])
def stop_solver():
    """Stop the solver"""
    global solver_running, token_send_running
    
    solver_running = False
    token_send_running = False
    generation_stats["status"] = "stopping"
    
    return jsonify({
        "message": "Stop signal sent",
        "tokens_generated": generation_stats["tokens_generated"],
        "tokens_sent": generation_stats.get("tokens_sent", 0)
    })

@app.route('/api/get-token', methods=['GET'])
def get_token():
    """Get a single token"""
    global tokens_cache
    
    get_new_tokens()
    
    with token_lock:
        if tokens_cache:
            token = tokens_cache.pop(0)
            return jsonify({
                "token": token,
                "remaining": len(tokens_cache),
                "total_generated": generation_stats["tokens_generated"]
            })
    
    return jsonify({"error": "No tokens available"}), 404

@app.route('/api/tokens', methods=['GET'])
def get_tokens():
    """Get multiple tokens"""
    global tokens_cache
    
    n = request.args.get('n', 10, type=int)
    n = min(n, 100)  # Increased max
    
    get_new_tokens()
    
    with token_lock:
        count = min(n, len(tokens_cache))
        result = tokens_cache[:count]
        tokens_cache = tokens_cache[count:]
        
        return jsonify({
            "tokens": result,
            "count": len(result),
            "remaining": len(tokens_cache),
            "total_generated": generation_stats["tokens_generated"]
        })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detailed statistics"""
    with stats_lock:
        if generation_stats.get("start_time"):
            start = datetime.fromisoformat(generation_stats["start_time"])
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed > 0:
                generation_stats["tokens_per_second"] = generation_stats["tokens_generated"] / elapsed
                generation_stats["uptime_seconds"] = int(elapsed)
    
    return jsonify({
        "status": generation_stats["status"],
        "tokens_generated": generation_stats["tokens_generated"],
        "tokens_available": len(tokens_cache),
        "threads": generation_stats["threads"],
        "start_time": generation_stats.get("start_time"),
        "tokens_per_second": round(generation_stats.get("tokens_per_second", 0), 2),
        "uptime_seconds": generation_stats.get("uptime_seconds", 0),
        "active_threads": threading.active_count(),
        "token_stats": {
            "sent": getattr(solver, 'token_stats', {}).get('sent', 0),
            "failed": getattr(solver, 'token_stats', {}).get('failed', 0),
            "batched": getattr(solver, 'token_stats', {}).get('batched', 0)
        }
    })

@app.route('/api/clear', methods=['POST'])
def clear_tokens():
    """Clear token cache"""
    global tokens_cache
    
    with token_lock:
        count = len(tokens_cache)
        tokens_cache = []
        
    return jsonify({
        "message": f"Cleared {count} tokens from cache",
        "remaining": 0
    })

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration"""
    global POOL_SIZE, TOKEN_BATCH_SIZE
    
    if request.method == 'GET':
        return jsonify({
            "pool_size": getattr(solver, 'POOL_SIZE', 200),
            "token_batch_size": getattr(solver, 'TOKEN_BATCH_SIZE', 100),
            "max_threads": 200,
            "current_threads": generation_stats["threads"],
            "solver_available": SOLVER_AVAILABLE
        })
    else:
        data = request.json or {}
        changes = {}
        
        if 'batch_size' in data:
            new_batch = min(data['batch_size'], 500)
            solver.TOKEN_BATCH_SIZE = new_batch
            changes['batch_size'] = new_batch
            
        return jsonify({
            "message": "Config updated",
            "changes": changes
        })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 6000))
    
    print(f"""
🔐 CN31 Solver - Ultra High Performance Edition
─────────────────────────────────────────────────
Port           : {port}
Max Threads    : 200 (configurable)
Default Threads: 100
Solver         : {'✅ Available' if SOLVER_AVAILABLE else '❌ Not Available'}
Files:
  - yidun_proxyless.py: {'✅' if os.path.exists('/app/yidun_proxyless.py') else '❌'}
  - dun163.js: {'✅' if os.path.exists('/app/dun163.js') else '❌'}
  - net.pkl: {'✅' if os.path.exists('/app/net.pkl') else '❌'}
""")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)