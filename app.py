from flask import Flask, render_template, request, jsonify
import random
from datetime import datetime

app = Flask(__name__)

class SimpleAnalyzer:
    def analyze_skin(self, image, user_data):
        # Simulate analysis
        features = {
            'pores': round(random.uniform(3, 8), 1),
            'redness': round(random.uniform(2, 7), 1),
            'texture': round(random.uniform(4, 9), 1),
            'spots': round(random.uniform(2, 6), 1),
            'wrinkles': round(random.uniform(2, 7), 1),
            'hydration': round(random.uniform(3, 8), 1),
            'evenness': round(random.uniform(4, 9), 1)
        }
        
        score = random.randint(40, 95)
        
        return {
            'score': score,
            'features': features,
            'recommendations': [
                "Use sunscreen daily",
                "Cleanse twice daily",
                "Moisturize regularly",
                "Drink plenty of water",
                "Get adequate sleep"
            ]
        }

analyzer = SimpleAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_skin():
    try:
        user_data = {
            'age': int(request.form.get('age', 30)),
            'skin_type': request.form.get('skin_type', 'combination'),
            'gender': request.form.get('gender', 'female')
        }
        
        analysis = analyzer.analyze_skin(None, user_data)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🚀 Skin Analyzer Started!")
    print("👉 Go to: http://localhost:5000")
    app.run(debug=True, port=5000)