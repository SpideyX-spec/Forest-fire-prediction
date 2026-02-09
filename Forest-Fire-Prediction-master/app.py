import pickle
import bz2
import os
import logging
from flask import Flask, request, render_template
import numpy as np

# --- 1. SETUP LOGGING ---
# We use standard logging to avoid errors if app_logger.py is broken
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- 2. ROBUST MODEL LOADER ---
def load_model(filename):
    """
    Tries to load a model whether it is BZ2 compressed or just a standard Pickle.
    """
    # Check if file exists first
    if not os.path.exists(filename):
        logger.error(f"CRITICAL ERROR: File {filename} not found!")
        return None

    try:
        # Try loading as BZ2 (Compressed)
        with bz2.BZ2File(filename, 'rb') as f:
            return pickle.load(f)
    except OSError:
        try:
            # If BZ2 fails, try loading as standard Pickle
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filename} as standard pickle: {e}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error loading {filename}: {e}")
        return None

# --- 3. LOAD THE MODELS ---
# Make sure these filenames match EXACTLY what is on GitHub (Case Sensitive!)
model_C = load_model('Classification.pkl')
model_R = load_model('Regression.pkl')

if model_C and model_R:
    logger.info("SUCCESS: Both models loaded correctly.")
else:
    logger.warning("WARNING: One or both models failed to load. Predictions will crash.")


# --- 4. ROUTES ---

@app.route('/')
def home():
    logger.info('Home page loaded successfully')
    return render_template('index.html')

@app.route('/predictC', methods=['POST'])
def predictC():
    if model_C is None:
        return render_template('index.html', prediction_text1="Error: Model not loaded")

    try:
        # Get data from form
        Temperature = float(request.form['Temperature'])
        Wind_Speed = float(request.form['Ws'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])

        features = [Temperature, Wind_Speed, FFMC, DMC, ISI]
        final_features = [np.array(features)]
        
        prediction = model_C.predict(final_features)[0]
        logger.info(f'Classification Prediction: {prediction}')

        if int(prediction) == 0:
            text = 'Forest is Safe!'
        else:
            text = 'Forest is in Danger!'

        return render_template('index.html', prediction_text1="{} --- Chance of Fire is {}".format(text, prediction))

    except Exception as e:
        logger.error(f"Error in Classification: {e}")
        return render_template('index.html', prediction_text1=f"Error: {e}")


@app.route('/predictR', methods=['POST'])
def predictR():
    if model_R is None:
        return render_template('index.html', prediction_text2="Error: Model not loaded")

    try:
        # Get data from form
        # CRITICAL: Ensure your HTML input names match these keys exactly!
        Temperature = float(request.form['Temperature'])
        
        # Note: Your HTML might use 'Wind_speed', check index.html name attribute!
        # If your HTML says name="Wind_speed", change 'Ws' below to 'Wind_speed'
        Wind_Speed = float(request.form.get('Ws', request.form.get('Wind_speed'))) 
        
        FFMC = float(request.form['FFMC'])
        
        # Handling potential naming mismatch for DMC/ISI based on your previous HTML
        DMC = float(request.form.get('DMC', request.form.get('DMC1')))
        ISI = float(request.form.get('ISI', request.form.get('ISI1')))

        features = [Temperature, Wind_Speed, FFMC, DMC, ISI]
        final_features = [np.array(features)]
        
        prediction = model_R.predict(final_features)[0]
        logger.info(f'Regression Prediction: {prediction}')

        if prediction > 15:
            res_text = "Fuel Moisture Code index is {:.4f} ---- Warning!!! High hazard rating".format(prediction)
        else:
            res_text = "Fuel Moisture Code index is {:.4f} ---- Safe.. Low hazard rating".format(prediction)
            
        return render_template('index.html', prediction_text2=res_text)

    except Exception as e:
        logger.error(f"Error in Regression: {e}")
        return render_template('index.html', prediction_text2=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
