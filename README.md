Project Title: Algerian Forest Fire Prediction System
Project Overview This is a full-stack Machine Learning web application designed to predict the risk of forest fires in Algeria based on meteorological data. The system utilizes two separate ML models to provide comprehensive risk analysis: a Classification model to predict if a fire will occur (Safe vs. Danger) and a Regression model to predict the intensity of the fire (Fire Weather Index).

Key Features

Dual Prediction System:

Classification: Uses Logistic Regression to classify the forest status as "Fire" or "Not Fire."

Regression: Uses Ridge Regression to predict the specific Fire Weather Index (FWI) value.

Interactive Web Interface: A responsive frontend built with HTML/CSS that allows users to input weather parameters (Temperature, Wind Speed, Humidity, etc.).

Cloud Database Integration: Connected to MongoDB Atlas to store dataset records and log application activity.

Deployment Ready: Configured with gunicorn and Procfile for deployment on platforms like Heroku or Render.

Tech Stack

Frontend: HTML5, CSS3, JavaScript.

Backend: Python, Flask Framework.

Machine Learning: Scikit-Learn, NumPy, Pandas (Models: Logistic Regression, Ridge).

Database: MongoDB (Pymongo).

Tools: VS Code, Git, Gunicorn.

How It Works

The user enters weather data (Temperature, RH, Ws, Rain, etc.) into the web form.

The Flask backend preprocesses the input using a Standard Scaler.

The pre-trained Machine Learning models (.pkl files) process the data.

The application returns a real-time prediction to the user interface indicating whether the forest is safe or at risk.
