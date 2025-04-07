#API exposing pre-trained Diabetes prediction model for eal-time inferencing.
# Kiran Veerabatheni -  @AppliedAIwithKiran
#--------------------------------------------
from flask import Flask, request,jsonify, send_from_directory
from flasgger import Swagger
import numpy as np
import pickle
import pandas as pd
import os
from flask_cors import CORS
import time 
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend to generate images without a GUI
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configure Swagger
app.config['SWAGGER'] = {
    'title': 'Diabetes Prediction API ',
    'description': 'An API for predicting diabetes using machine learning.',
    'version': '1.0.0',
    'uiversion': 3
}
swagger = Swagger(app)

#load the model and scaler
with open(os.path.join('..', 'models', 'k_random_forest_model.pkl'), 'rb') as k_model_file:
    k_model = pickle.load(k_model_file)
with open(os.path.join('..', 'models', 'k_scaler.pkl'), 'rb') as k_scaler_file:
    k_scaler = pickle.load(k_scaler_file)
#get the trainig data from the model for plotting 
with open(os.path.join('..', 'models', 'k_training_data.pkl'), 'rb') as f:
    training_data = pickle.load(f)
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    feature_names = training_data['feature_names']

@app.route('/')
def home():
    """
    Home page for the Diabetes Prediction API.
    ---
    tags:
      - Home
    responses:
      200:
        description: Returns a simple HTML page with API instructions.
    """
    return """
    <h1>Diabetes Prediction API</h1>
    <p>An API for predicting diabetes using machine learning.</p>
    <p>Send a POST request to /predict with the following features:</p>
    <ul>
        <li>Pregnancies</li>
        <li>Glucose</li>
        <li>BloodPressure</li>
        <li>SkinThickness</li>
        <li>Insulin</li>
        <li>BMI</li>
        <li>DiabetesPedigreeFunction</li>
        <li>Age</li>
    </ul>
    """

# Feature names 
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]


# Define the path to the 'public' directory in the React app
#VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'react', 'k-diabetes-web', 'public')
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict diabetes risk based on patient data.
    ---
    tags:
      - Predictions
    parameters:
      - name: body
        in: body
        required: true
        description: Patient data for diabetes prediction
        schema:
          type: object
          required:
            - Pregnancies
            - Glucose
            - BloodPressure
            - SkinThickness
            - Insulin
            - BMI
            - DiabetesPedigreeFunction
            - Age
          properties:
            Pregnancies:
              type: number
              description: Number of times pregnant
              example: 6
            Glucose:
              type: number
              description: Plasma glucose concentration
              example: 148
            BloodPressure:
              type: number
              description: Diastolic blood pressure (mm Hg)
              example: 72
            SkinThickness:
              type: number
              description: Triceps skin fold thickness (mm)
              example: 35
            Insulin:
              type: number
              description: 2-Hour serum insulin (mu U/ml)
              example: 0
            BMI:
              type: number
              description: Body mass index
              example: 33.6
            DiabetesPedigreeFunction:
              type: number
              description: Diabetes pedigree function
              example: 0.627
            Age:
              type: number
              description: Age in years
              example: 50
    responses:
      200:
        description: Prediction result with visualization URLs
        schema:
          type: object
          properties:
            status:
              type: string
              description: Processing status
              example: "complete"
            prediction:
              type: integer
              description: Binary prediction (0=Negative, 1=Positive)
              example: 1
            probability:
              type: number
              description: Probability of diabetes (0-1)
              example: 0.85
            diabetes_status:
              type: string
              description: Human-readable prediction status
              example: "Positive"
            model_used:
              type: string
              description: Model used for prediction
              example: "Random Forest"
            visualizations:
              type: object
              description: URLs to generated visualization images
              properties:
                radar_chart:
                  type: string
                  example: "prediction_radar.png"
                scatter_plot:
                  type: string
                  example: "prediction_scatter.png"
                pie_chart:
                  type: string
                  example: "prediction_pie.png"
                donut_chart:
                  type: string
                  example: "prediction_donut.png"
                bar_chart: 
                  type: string
                  example: "prediction_bar.png"
                box_plot: 
                  type: string
                  example: "prediction_boxplot.png"
            completion:
              type: object
              description: Visualization completion status
              properties:
                radar:
                  type: boolean
                  example: true
                scatter:
                  type: boolean
                  example: true
                pie:
                  type: boolean
                  example: true
                donut:
                  type: boolean
                  example: true
                bar:
                  type: boolean
                  example: true
                box:
                  type: boolean
                  example: true
      400:
        description: Invalid input
        schema:
          type: object
          properties:
            status:
              type: string
              example: "error"
            error:
              type: string
              example: "Missing required feature: Glucose"
      500:
        description: Server error
        schema:
          type: object
          properties:
            status:
              type: string
              example: "error"
            message:
              type: string
              example: "Error processing request"
            partial_data:
              type: object
              description: "Partial results if available"
    """
    try:
        # Process input data
        data = request.get_json(force=True)
        print("Received data:", data)
        
        # Validate required fields
        required_features = ['Pregnancies','Glucose','BloodPressure', 'SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction','Age']
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    'status': 'error',
                    'error': f'Missing required feature: {feature}'
                }), 400
        
        # Prepare input data
        in_data = pd.DataFrame([[
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]], columns=required_features)
        
        # Scale and predict
        in_data_scaled = k_scaler.transform(in_data)
        prediction = k_model.predict(in_data_scaled)[0]
        prediction_proba = k_model.predict_proba(in_data_scaled)[0][1]
        
        # Initialize result
        result = {
            'status': 'processing',
            'prediction': int(prediction),
            'probability': float(prediction_proba),
            'diabetes_status': 'Positive' if prediction == 1 else 'Negative',
            'model_used': 'Random Forest',
            'visualizations': {},
            'completion': {
                'radar': False, 'scatter': False, 'pie': False, 'donut': False, 'bar': False, 'box': False
            }
        }
        
        # Ensure visualization directory exists
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        # Generate visualizations - each wrapped in verification
        visualization_functions = [
            ('radar', generate_radar_chart),
            ('scatter', generate_scatter_plot),
            ('pie', generate_pie_chart),
            ('donut', generate_donut_chart),
            ('bar', generate_bar_chart),
            ('box', generate_box_plot)
        ]
        
        for viz_type, viz_func in visualization_functions:
            filename = os.path.join(VISUALIZATION_DIR, f'prediction_{viz_type}.png')
            try:
                viz_func(result, in_data, in_data_scaled, filename)
                if verify_file(filename):
                    result['visualizations'][f'{viz_type}_chart'] = f'prediction_{viz_type}.png'
                    result['completion'][viz_type] = True
                else:
                    raise Exception(f"Failed to verify {viz_type} chart")
            except Exception as e:
                result['visualizations'][f'{viz_type}_error'] = str(e)
        # Verify all visualizations completed
        if not all(result['completion'].values()):
            missing = [k for k,v in result['completion'].items() if not v]
            raise Exception(f"Visualizations incomplete: {missing}")
        result['status'] = 'complete'
        return jsonify(result)
        
    except Exception as err:
        return jsonify({
            'status': 'error',
            'message': str(err),
            'partial_data': result if 'result' in locals() else None
        }), 500


# Helper function to verify file creation
def verify_file(filename, retries=3, delay=1):
    for _ in range(retries):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            return True
        time.sleep(delay)
    return False


def generate_radar_chart(result, in_data, in_data_scaled, output_path):
    """Generate radar chart comparing current prediction to training data averages"""
    try:
        labels = ['Probability', 'Diabetes Status', 'Confidence', 'Risk Level']
        
        # Calculate average values from training data
        train_probas = k_model.predict_proba(X_train)[:, 1]
        avg_train_prob = np.mean(train_probas)
        pos_rate = np.mean(y_train)
        
        # Values for current prediction
        current_values = [
            result['probability'], 
            1 if result['diabetes_status'] == 'Positive' else 0, 
            0.9,  # Confidence
            0.7   # Risk Level
        ]
        
        # Values for training data averages
        train_values = [
            avg_train_prob,
            pos_rate,
            0.8,  # Average confidence
            0.6   # Average risk
        ]
        
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot training data
        train_values += train_values[:1]
        ax.plot(angles, train_values, color='blue', linewidth=1, label='Training Average')
        ax.fill(angles, train_values, color='blue', alpha=0.1)
        
        # Plot current prediction
        current_values += current_values[:1]
        ax.plot(angles, current_values, color='red', linewidth=1, label='Current Prediction')
        ax.fill(angles, current_values, color='red', alpha=0.1)
        
        # Add labels and legend
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.legend(loc='upper right')
        plt.title('Diabetes Risk Comparison Radar Chart', pad=20)
        
        plt.savefig(output_path, bbox_inches='tight', dpi=120)
        plt.close()
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def generate_scatter_plot(result, in_data, in_data_scaled, output_path):
    """Generate scatter plot of probabilities vs glucose levels"""
    try:
        # Get probabilities for all training points
        train_probas = k_model.predict_proba(X_train)[:, 1]
        
        plt.figure(figsize=(12, 8))
        
        # Plot all training data points
        plt.scatter(
            x=train_probas,
            y=X_train[:, 1],  # Glucose values
            c=['blue' if x == 0 else 'brown' for x in y_train],
            alpha=0.4,
            label='Training Data',
            s=30
        )
        
        # Plot current prediction point
        plt.scatter(
            x=[result['probability']],
            y=[in_data_scaled[0, 1]],  # Scaled glucose value
            s=400,
            c=['green' if result['diabetes_status'] == 'Negative' else 'red'],
            edgecolors='gold',
            linewidth=1,
            label=f'Current Prediction ({result["diabetes_status"]})',
            marker='*'
        )
        
        # Add reference lines and styling
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.text(0.5, np.max(X_train[:, 1])*0.95, 'Decision Boundary', 
                rotation=90, va='top', ha='right', color='gray')
        
        plt.title('Diabetes Prediction: Probabilities vs Glucose Levels')
        plt.xlabel('Predicted Probability of Diabetes')
        plt.ylabel('Glucose Level (Standardized)')
        plt.xlim(-0.05, 1.05)
        plt.ylim(np.min(X_train[:, 1])-0.5, np.max(X_train[:, 1])+0.5)
        plt.grid(True, alpha=0.2)
        plt.legend(loc='upper right')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=120)
        plt.close()
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def generate_pie_chart(result, in_data, in_data_scaled, output_path):
    """Generate pie chart of prediction probabilities"""
    try:
        plt.figure(figsize=(8, 8))
        sizes = [result['probability'], 1 - result['probability']]
        colors = ['lightcoral', 'lightskyblue']
        explode = (0.1, 0)  # Explode the prediction slice
        
        plt.pie(
            sizes, 
            explode=explode, 
            labels=['Diabetes', 'No Diabetes'], 
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=140
        )
        plt.title('Diabetes Prediction Probability')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def generate_donut_chart(result, in_data, in_data_scaled, output_path):
    """Generate donut chart of prediction probabilities"""
    try:
        plt.figure(figsize=(8, 8))
        sizes = [result['probability'], 1 - result['probability']]
        colors = ['#ff9999','#66b3ff']
        
        plt.pie(
            sizes, 
            colors=colors, 
            labels=['Positive', 'Negative'],
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(width=0.4, edgecolor='w')
        )
        
        # Draw white circle in center
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.title('Diabetes Prediction Donut Chart')
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def generate_bar_chart(result, in_data, in_data_scaled, output_path):
    """Generate horizontal bar chart of feature importances"""
    try:
        # Get feature importances
        if hasattr(k_model, 'feature_importances_'):
            importances = k_model.feature_importances_
        else:
            # For models without feature_importances_, use coefficients
            if hasattr(k_model, 'coef_'):
                importances = np.abs(k_model.coef_[0])
            else:
                importances = np.ones(len(feature_names)) / len(feature_names)
        
        # Create sorted DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        
        # Plot all features
        bars = plt.barh(
            feature_importance['Feature'],
            feature_importance['Importance'],
            color='skyblue'
        )
        
        # Highlight most important feature for this prediction
        max_feature = feature_importance.iloc[-1]['Feature']
        max_idx = feature_importance[feature_importance['Feature'] == max_feature].index[0]
        bars[max_idx].set_color('orange')
        
        # Add value annotations
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'], 
                                                     feature_importance['Importance'])):
            value = in_data[feature].values[0]
            plt.text(
                importance + 0.01,
                i,
                f'Value: {value:.1f}',
                va='center',
                fontsize=9
            )
        
        plt.title('Feature Importance for Diabetes Prediction')
        plt.xlabel('Relative Importance')
        plt.grid(axis='x', alpha=0.3)
        
        plt.savefig(output_path, bbox_inches='tight', dpi=120)
        plt.close()
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def generate_box_plot(result, in_data, in_data_scaled, output_path):
    """Generate box plots comparing patient values to training data"""
    try:
        # Select top 4 most important features
        if hasattr(k_model, 'feature_importances_'):
            top_features_idx = np.argsort(k_model.feature_importances_)[-4:]
        else:
            top_features_idx = range(min(4, len(feature_names)))
        
        plt.figure(figsize=(14, 8))
        
        for i, idx in enumerate(top_features_idx):
            feature = feature_names[idx]
            plt.subplot(2, 2, i+1)
            
            # Plot training data distribution
            train_data = X_train[:, idx]
            
            box = plt.boxplot(
                [train_data[y_train == 0], train_data[y_train == 1]],
                patch_artist=True,
                tick_labels=['Negative', 'Positive'],  # Changed from 'labels' to 'tick_labels'
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='black')
            )
            
            # Plot current patient's value
            patient_value = in_data_scaled[0, idx]
            plt.scatter(
                [1, 2],  # Positions for both boxes
                [patient_value, patient_value],
                color='red',
                marker='*',
                s=200,
                label='Current Patient'
            )
            
            # Add feature name and units
            units = {
                'Glucose': '(mg/dL)',
                'BloodPressure': '(mm Hg)',
                'BMI': '(kg/m²)',
                'Age': '(years)'
            }.get(feature, '')
            
            plt.title(f'{feature} {units} Distribution')
            plt.ylabel('Standardized Value')
            
            # Add reference line
            plt.axhline(
                y=patient_value,
                color='red',
                linestyle='--',
                alpha=0.5
            )
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=120)
        plt.close()
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    


@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get information about the model k-diabetes-prediction.
    ---
    tags:
      - Model Information
    responses:
      200:
        description: Information about the model
        schema:
          type: object
          properties:
            model_name:
              type: string
              example: Random Forest Classifier
            features:
              type: array
              items:
                type: string
              example: ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            target:
              type: string
              example: Diabetes (1 = Positive, 0 = Negative)
    """
    return jsonify({
        'model_name': 'Random Forest Classifier',
        'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'target': 'Diabetes (1 = Positive, 0 = Negative)'
    })
 

# Define the path to the 'public' directory in the React app
#VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'react', 'k-diabetes-predictor-web', 'public')
#VISUALIZATION_DIR = r"C:\Kiran\TRAINING\Diabetes-Detection\react\k-diabetes-web\public"
# Create the directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
# Serve the 'visualization' directory as a static folder
app.static_folder = VISUALIZATION_DIR
#app.static_url_path = '/visualizations'

@app.route('/images', methods=['GET'])
def list_images():
    """
    Get a list of all images in the 'visualization' directory with their URLs.
    ---
    tags:
      - Images
    responses:
      200:
        description: A list of image URLs in the 'visualization' directory.
        schema:
          type: object
          properties:
            images:
              type: array
              items:
                type: string
              example: ["http://127.0.0.1:5500/visualization/k_feature_distributions.png"]
    """
    try:
        # Check if the directory exists
        if not os.path.exists(VISUALIZATION_DIR):
            return jsonify({'error': f'Directory "{VISUALIZATION_DIR}" does not exist'}), 404

        # Get a list of all files in the directory
        files = os.listdir(VISUALIZATION_DIR)

        # Filter for image files (common extensions: .png, .jpg, .jpeg, .gif, .bmp, .svg)
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
        images = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]

        # Construct full URLs for the images
        base_url = request.host_url.rstrip('/')  # Get the base URL (e.g., http://127.0.0.1:5500)
        image_urls = [f"{base_url}/visualization/{image}" for image in images]

        # Return the list of image URLs
        return jsonify({'images': image_urls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Configure CORS more broadly - simplified approach
CORS(app, resources={r"/*": {"origins": "*"}})

# Allow ONLY your React frontend (running on http://localhost:3000 or your production URL)
# CORS(app, resources={
#     r"/*": {
#         "origins": ["http://192.168.1.20:3000", "https://your-production-domain.com"],
#         "methods": ["GET", "POST", "OPTIONS"],  # Allowed HTTP methods
#         "allow_headers": ["Content-Type"]  # Allowed headers
#     }
# })

# Define the path to your images directory
IMAGE_FOLDER = os.path.join(os.getcwd(), "img")

@app.route('/img/<filename>')
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8800)

# Production server launch with Waitress
if __name__ == '__main__':
    from waitress import serve
    serve(
        app,
        host="0.0.0.0",  
        port=8800,        
        threads=4         
    )