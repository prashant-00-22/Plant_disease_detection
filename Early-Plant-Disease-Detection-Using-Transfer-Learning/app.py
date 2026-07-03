"""
Plant Disease Detection Web Application
Using Flask and MobileNetV2 Transfer Learning Model

This application provides a web interface for detecting plant diseases
from leaf images using a pre-trained MobileNetV2 model.
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from PIL import Image
try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'plant_disease_detection_secret_key'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/plant_disease_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease classes (based on theCorn dataset)
# Update these based on your actual model classes
DISEASE_CLASSES = {
    0: 'Healthy',
    1: 'Northern Leaf Blight',
    2: 'Common Rust',
    3: 'Cercospora Leaf Spot (Gray Leaf Spot)'
}

# Additional plant types if using full dataset
PLANT_CLASSES = ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 
                 'Peach', 'Pepper', 'Potato', 'Raspberry', 'Soybean', 
                 'Squash', 'Strawberry', 'Tomato']

# Global model variable (loaded lazily / safely)
model = None
model_load_attempted = False
model_load_error = None


def load_model():
    """Load the trained model safely.

    Requirement: server must never crash on startup due to missing TensorFlow/model.
    """
    global model, model_load_attempted, model_load_error

    # Prevent repeated expensive attempts
    if model_load_attempted:
        return

    model_load_attempted = True
    model_load_error = None

    try:
        # TensorFlow might not be importable in this environment
        if tf is None:
            raise RuntimeError("TensorFlow is not available (tf import failed).")

        if not os.path.exists(MODEL_PATH):
            print(f"Model not found at {MODEL_PATH}")
            raise FileNotFoundError(
                f"Model file missing at {MODEL_PATH}. Please place the model in the models folder."
            )

        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        # Log actual exception for debugging
        model_load_error = repr(e)
        print(f"Error loading model: {e}")
        print("--- load_model exception traceback follows ---")
        import traceback

        traceback.print_exc()

        # Keep server running; prediction will return JSON error
        model = None


def create_demo_model():
    """Create a demo model for testing when no trained model is available"""
    if tf is None:
        raise RuntimeError("TensorFlow is not available in this environment")

    # This creates a simple model for demonstration
    # In production, you would use the trained MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    predictions = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    global tf

    if tf is None:
        print("preprocess_image: TensorFlow is not available")
        return None

    try:
        # Load and resize image
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure RGB mode
        img = img.resize((224, 224))

        # Convert to array
        img_array = np.array(img)

        # Preprocess for MobileNetV2
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Expand dimensions to match model input
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        # Log actual exception for debugging
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_disease(image_path):
    """Predict disease from the image.

    If model is unavailable, this function returns (None, None, 0) and logs.
    """
    global model

    if model is None:
        # Model unavailable - do not crash
        print("predict_disease: model is unavailable (model is None)")
        return None, None, 0

    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)

        if processed_img is None:
            return None, None, 0

        # Make prediction
        predictions = model.predict(processed_img, verbose=0)

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Get disease name
        disease_name = DISEASE_CLASSES.get(predicted_class, "Unknown")

        # Get all class probabilities
        all_predictions = {
            DISEASE_CLASSES[i]: float(predictions[0][i]) * 100
            for i in range(len(predictions[0]))
        }

        return disease_name, all_predictions, float(confidence) * 100
    except Exception as e:
        # Log actual exception for debugging
        print(f"Error predicting disease: {e}")
        import traceback

        traceback.print_exc()
        return None, None, 0


# ==================== ROUTES ====================

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload page - handles both GET and POST requests"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Redirect to result page with the image path
            return redirect(url_for('result', image=filename))
    
    return render_template('upload.html')

@app.route('/result')
def result():
    """Result page - displays prediction results"""
    image_name = request.args.get('image')

    if not image_name:
        flash('No image provided', 'error')
        return redirect(url_for('upload'))

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

    if not os.path.exists(image_path):
        flash('Image not found', 'error')
        return redirect(url_for('upload'))

    # Make prediction
    disease_name, all_predictions, confidence = predict_disease(image_path)

    if disease_name is None:
        # If the model is unavailable, return a clear JSON error (HTTP 500)
        if model is None:
            return {
                "error": "Model unavailable",
                "message": "The prediction model could not be loaded. Please train the model and place it at models/plant_disease_model.h5, or reinstall dependencies.",
            }, 500

        flash('Error processing image', 'error')
        return redirect(url_for('upload'))

    return render_template(
        'result.html',
        image=image_name,
        disease=disease_name,
        confidence=confidence,
        all_predictions=all_predictions
    )


@app.route('/how-it-works')
def how_it_works():
    """How it works page"""
    return render_template('how_it_works.html')

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def page_not_found(e):
    """404 error handler"""
    return render_template('error.html', error_code=404, message="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """500 error handler"""
    return render_template('error.html', error_code=500, message="Internal server error"), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    # Load model on startup (must not crash the server)
    try:
        load_model()
    except Exception as e:
        # Log exception and continue serving pages
        import traceback
        print(f"Startup model loading failed: {e}")
        traceback.print_exc()
    
    # Run the app

    print("=" * 50)
    print("Plant Disease Detection System")
    print("=" * 50)
    print("Starting server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
