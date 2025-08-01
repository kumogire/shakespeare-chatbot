"""
Shakespeare Chatbot - Web Interface
A Flask-based web interface for chatting with the trained Shakespeare model.
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_utils import ShakespeareModel

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')
CORS(app)

# Global model instance
shakespeare_model = None

def initialize_model():
    """Initialize the Shakespeare model"""
    global shakespeare_model
    try:
        model_path = os.environ.get('MODEL_PATH', 'models/shakespeare_model')
        shakespeare_model = ShakespeareModel(model_path)
        print("✅ Shakespeare model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you've trained the model first with 'make train'")
        return False

@app.route('/')
def home():
    """Serve the main chat interface"""
    return render_template('chat.html')

@app.route('/health')
def health():
    """Health check endpoint for Docker"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': shakespeare_model is not None
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and return bot responses"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        if shakespeare_model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Generate response using the model
        bot_response = shakespeare_model.generate_response(
            user_message, 
            max_length=data.get('max_length', 100),
            temperature=data.get('temperature', 0.8)
        )
        
        return jsonify({
            'response': bot_response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample')
def sample():
    """Generate a sample text from the model"""
    try:
        if shakespeare_model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        sample_text = shakespeare_model.generate_sample(max_length=200)
        return jsonify({
            'sample': sample_text,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Get system status"""
    return jsonify({
        'model_loaded': shakespeare_model is not None,
        'model_path': os.environ.get('MODEL_PATH', 'models/shakespeare_model'),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update generation configuration"""
    if request.method == 'GET':
        if shakespeare_model:
            return jsonify(shakespeare_model.get_config())
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            if shakespeare_model:
                shakespeare_model.update_config(data)
                return jsonify({'status': 'success'})
            else:
                return jsonify({'error': 'Model not loaded'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to run the Flask app"""
    print("🎭 Starting Shakespeare Chatbot Interface...")
    
    # Initialize model
    if not initialize_model():
        print("⚠️  Warning: Model not loaded. Some features may not work.")
        print("   Run 'make train' to train the model first.")
    
    # Get configuration from environment
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🌐 Starting server on http://{host}:{port}")
    print(f"🎭 Open your browser and start chatting with Shakespeare!")
    print(f"💡 Try messages like:")
    print(f"   - 'How art thou today?'")
    print(f"   - 'Tell me about love'")
    print(f"   - 'What think you of the weather?'")
    print()
    print(f"🛑 Press Ctrl+C to stop the server")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Shakespeare Chatbot. Farewell!")

if __name__ == '__main__':
    main()