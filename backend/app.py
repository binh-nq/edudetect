from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_engine import InferenceEngine
from rewrite_engine import RewriteEngine
from config import Config
import sys

app = Flask(__name__)
CORS(app)

engine = None
rewrite_engine = None

def initialize_engine():
    global engine, rewrite_engine
    
    try:

        Config.validate_model_path()
        model_path = Config.MODEL_PATH
        
        print("\n" + "=" * 60)
        print("Initializing AI Text Detection Engine")
        print("=" * 60)

        engine = InferenceEngine(model_path=model_path)
        
        print("=" * 60)
        print("[OK] Detection engine ready")
        print("=" * 60 + "\n")
        
        try:
            print("=" * 60)
            print("Initializing Rewrite Engine")
            print("=" * 60)
            rewrite_engine = RewriteEngine()
            print("=" * 60)
            print("[OK] Rewrite engine ready")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"[WARNING] Rewrite engine not loaded: {e}")
            print("Rewrite functionality will be disabled.\n")
        
        return True
        
    except (ValueError, FileNotFoundError) as e:
        print("\n" + "=" * 60)
        print("ERROR: Failed to initialize server")
        print("=" * 60)
        print(f"\n{str(e)}\n")
        print("=" * 60)
        print("Server cannot start without a valid fine-tuned model.")
        print("=" * 60 + "\n")
        return False
    
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Unexpected error during initialization")
        print("=" * 60)
        print(f"\n{str(e)}\n")
        print("=" * 60 + "\n")
        return False

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if engine is None:
        return jsonify({
            'error': 'Server not properly initialized. Model not loaded.'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in request body'
            }), 400
        
        text = data['text']
        
        # Analyze
        result = engine.analyze(text)
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/rewrite', methods=['POST'])
def rewrite():
    if rewrite_engine is None:
        return jsonify({
            'error': 'Rewrite engine not available. Model not loaded.'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'target' not in data:
            return jsonify({
                'error': 'Missing "target" field in request body'
            }), 400
        
        mode = data.get('mode', 'fix')
        target = data['target']
        prev_context = data.get('prev_context', '')
        next_context = data.get('next_context', '')
        
        # Rewrite
        rewritten = rewrite_engine.rewrite(
            target=target,
            mode=mode,
            prev_context=prev_context,
            next_context=next_context
        )
        
        return jsonify({'rewritten': rewritten}), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        print(f"Error during rewriting: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if engine is None:
        return jsonify({
            'status': 'error',
            'message': 'Detection model not loaded'
        }), 503
    
    return jsonify({
        'status': 'ok',
        'detect_model': Config.MODEL_PATH,
        'rewrite_available': rewrite_engine is not None
    }), 200

if __name__ == '__main__':
    if not initialize_engine():
        print("FATAL: Cannot start server without model.")
        print("\nUsage:")
        print("  export MODEL_PATH=/path/to/your/fine-tuned-phobert")
        print("  python app.py")
        sys.exit(1)
    
    app.run(host='0.0.0.0', port=5000, debug=True)