import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import math
import platform
import sys
import warnings
import pytz  # Added for timezone support
import time  # Added for timezone setting

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Set timezone to IST (Indian Standard Time)
os.environ['TZ'] = 'Asia/Kolkata'
try:
    time.tzset()  # This works on Render (Unix-based systems)
except AttributeError:
    pass  # For Windows local development

# Model path
MODEL_PATH = os.path.join('model', 'delhi_travel_time_model.pkl')

# Dictionary of Delhi locations with coordinates
DELHI_LOCATIONS = {
    'Connaught Place': {'lat': 28.6289, 'lon': 77.2088},
    'India Gate': {'lat': 28.6129, 'lon': 77.2295},
    'Chandni Chowk': {'lat': 28.6527, 'lon': 77.2304},
    'Karol Bagh': {'lat': 28.6517, 'lon': 77.1909},
    'South Extension': {'lat': 28.5678, 'lon': 77.2199},
    'Lajpat Nagar': {'lat': 28.5677, 'lon': 77.2426},
    'Saket': {'lat': 28.5245, 'lon': 77.2062},
    'Dwarka': {'lat': 28.5921, 'lon': 77.0460},
    'Rohini': {'lat': 28.7345, 'lon': 77.0825},
    'Noida': {'lat': 28.5355, 'lon': 77.3910},
    'Gurgaon': {'lat': 28.4595, 'lon': 77.0266},
    'Delhi Airport': {'lat': 28.5562, 'lon': 77.1000},
    'New Delhi Railway Station': {'lat': 28.6424, 'lon': 77.2206},
    'Old Delhi Railway Station': {'lat': 28.6667, 'lon': 77.2278},
    'Kashmere Gate': {'lat': 28.6689, 'lon': 77.2291},
    'Rajouri Garden': {'lat': 28.6520, 'lon': 77.1215},
    'Janakpuri': {'lat': 28.6219, 'lon': 77.0835},
    'Vasant Kunj': {'lat': 28.5433, 'lon': 77.1553},
    'Pitampura': {'lat': 28.6959, 'lon': 77.1479}
}

# Try to load the trained model (if it exists)
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        print("✅ Using fallback calculation logic")
else:
    print("ℹ️ Model file not found. Using fallback calculation logic")

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2) * math.sin(delta_lat/2) + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * \
        math.sin(delta_lon/2) * math.sin(delta_lon/2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def get_traffic_multiplier(traffic_level):
    """Get multiplier based on traffic level"""
    traffic_multipliers = {
        'low': 1.0,
        'medium': 1.5,
        'high': 2.2,
        'very_high': 3.0
    }
    return traffic_multipliers.get(traffic_level, 1.5)

def get_weather_multiplier(weather_condition):
    """Get multiplier based on weather condition"""
    weather_multipliers = {
        'clear': 1.0,
        'rain': 1.3,
        'fog': 1.4,
        'storm': 1.6
    }
    return weather_multipliers.get(weather_condition, 1.0)

def predict_travel_time(start_location, end_location, vehicle_speed, traffic_level, weather_condition):
    """
    Predict travel time based on inputs
    Uses ML model if available, otherwise falls back to mathematical calculation
    """
    
    # Get coordinates
    start_coords = DELHI_LOCATIONS.get(start_location)
    end_coords = DELHI_LOCATIONS.get(end_location)
    
    if not start_coords or not end_coords:
        raise ValueError("Invalid location selected")
    
    # Calculate distance
    distance = calculate_distance(
        start_coords['lat'], start_coords['lon'],
        end_coords['lat'], end_coords['lon']
    )
    
    # Get multipliers
    traffic_mult = get_traffic_multiplier(traffic_level)
    weather_mult = get_weather_multiplier(weather_condition)
    
    # Validate speed
    if vehicle_speed <= 0 or vehicle_speed > 120:
        vehicle_speed = 30  # default to 30 km/h if invalid
    
    # Calculate using ML model if available
    if model is not None:
        try:
            # Prepare features for the model
            # Adjust these features based on what your model was trained on
            features = np.array([[
                distance,           # Distance in km
                vehicle_speed,      # Speed in km/h
                traffic_mult,       # Traffic multiplier
                weather_mult        # Weather multiplier
            ]])
            
            # Make prediction using the model
            predicted_time_hours = model.predict(features)[0]
            print("✅ Used ML model for prediction")
        except Exception as e:
            print(f"⚠️ Model prediction failed: {e}, using fallback")
            # Fallback to mathematical calculation
            base_time_hours = distance / vehicle_speed
            predicted_time_hours = base_time_hours * traffic_mult * weather_mult
    else:
        # Mathematical calculation fallback
        base_time_hours = distance / vehicle_speed
        predicted_time_hours = base_time_hours * traffic_mult * weather_mult
        print("ℹ️ Used mathematical calculation")
    
    # Ensure prediction is reasonable
    predicted_time_hours = max(0.1, predicted_time_hours)  # Minimum 6 minutes
    
    # Convert to minutes
    predicted_time_minutes = predicted_time_hours * 60
    
    # FIXED: Use IST timezone (Indian Standard Time) instead of UTC
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    arrival_time = current_time + timedelta(minutes=predicted_time_minutes)
    
    # Format traffic level for display
    traffic_display = traffic_level.replace('_', ' ').title()
    
    return {
        'distance_km': round(distance, 2),
        'travel_time_minutes': round(predicted_time_minutes, 1),
        'travel_time_hours': round(predicted_time_hours, 2),
        'arrival_time': arrival_time.strftime('%I:%M %p'),
        'arrival_date': arrival_time.strftime('%d %b %Y'),
        'traffic_level': traffic_display,
        'weather_condition': weather_condition.title(),
        'average_speed': vehicle_speed,
        'start_location': start_location,
        'end_location': end_location
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', 
                         locations=sorted(DELHI_LOCATIONS.keys()),
                         traffic_levels=['low', 'medium', 'high', 'very_high'],
                         weather_conditions=['clear', 'rain', 'fog', 'storm'])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        start_location = request.form['start_location']
        end_location = request.form['end_location']
        vehicle_speed = float(request.form['vehicle_speed'])
        traffic_level = request.form['traffic_level']
        weather_condition = request.form['weather_condition']
        
        # Validate inputs
        if not start_location or not end_location:
            return jsonify({
                'success': False,
                'error': 'Please select both start and end locations'
            }), 400
        
        if start_location == end_location:
            return jsonify({
                'success': False,
                'error': 'Start and end locations must be different'
            }), 400
        
        if vehicle_speed <= 0 or vehicle_speed > 120:
            return jsonify({
                'success': False,
                'error': 'Vehicle speed must be between 1 and 120 km/h'
            }), 400
        
        # Make prediction
        result = predict_travel_time(
            start_location, end_location,
            vehicle_speed, traffic_level, weather_condition
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        start_location = data.get('start_location')
        end_location = data.get('end_location')
        vehicle_speed = float(data.get('vehicle_speed', 30))
        traffic_level = data.get('traffic_level', 'medium')
        weather_condition = data.get('weather_condition', 'clear')
        
        if not start_location or not end_location:
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        result = predict_travel_time(
            start_location, end_location,
            vehicle_speed, traffic_level, weather_condition
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/locations')
def get_locations():
    """Return list of available locations"""
    return jsonify({
        'success': True,
        'locations': sorted(DELHI_LOCATIONS.keys()),
        'count': len(DELHI_LOCATIONS)
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    # Get current time in IST for verification
    ist = pytz.timezone('Asia/Kolkata')
    current_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    return jsonify({
        'status': 'healthy',
        'python_version': sys.version.split()[0],
        'platform': platform.machine(),
        'model_loaded': model is not None,
        'locations_count': len(DELHI_LOCATIONS),
        'current_time_ist': current_ist,
        'timezone': 'Asia/Kolkata',
        'message': 'Delhi Travel Time Predictor is running'
    })

if __name__ == '__main__':
    # Get current time in IST for startup message
    ist = pytz.timezone('Asia/Kolkata')
    current_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "="*60)
    print("🚗 DELHI TRAVEL TIME PREDICTOR")
    print("="*60)
    print(f"📅 Started at: {current_ist} IST")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"💻 Platform: {platform.machine()} ({platform.system()})")
    print(f"📍 Locations loaded: {len(DELHI_LOCATIONS)}")
    print(f"🤖 ML Model: {'✅ Loaded' if model else '⚠️ Using fallback'}")
    print(f"⏰ Timezone: Asia/Kolkata (IST)")
    print("="*60)
    print("\n🌐 Server starting...")
    print("📱 Access the app:")
    print("   → Local: http://localhost:5000")
    if 'RENDER' in os.environ:
        print("   → Render: https://delhi-travel-predictor.onrender.com")
    print("\n🛑 Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Get port from environment variable (for Render) or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
