from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify ,g
import sqlite3
import numpy as np
import pickle
import os
import jwt
from datetime import datetime, timedelta
from config import Config
from models import init_db, get_db
import logging
from dotenv import load_dotenv
import joblib
import traceback
import pandas as pd 
import crypten
load_dotenv()
from werkzeug.security import generate_password_hash, check_password_hash
import crypten

import crypten.nn as nn

import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
PORT = int(os.getenv('PORT', 5000))

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
class EncryptedLogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(EncryptedLogisticRegression, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x).sigmoid()  # Sigmoid activation for binary classification

model_1 = None
scaler_1= None
model_2 = None
scaler_2= None
try:
    crypten.init()
    model_1 = joblib.load('encrypted_model_1_concrete.joblib')
    scaler_1 = joblib.load('encrypted_model_1_scalar.joblib')
    with open('encrypted_model_2_crypten.pkl', 'rb') as f:
        model_2 = pickle.load(f)

    scaler_2 = joblib.load('encrypted_model_2_scalar.pkl')
    logging.info("Models and scaler loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Error loading models or scaler: {e}")
    print(" One or more model files are missing. Please check the file paths.")
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    print(" An unexpected error occurred while loading models.")
    traceback.print_exc()  

if model_1 is None  or scaler_1 is None:
    print(" Some models were not loaded properly. Please retrain or check paths.")
    traceback.print_exc()  
else:
    print(" Models and scaler loaded successfully.")
from flask import g

@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.current_user = None
    else:
        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE id = ?', (user_id,)
        ).fetchone()

        if user:
            g.current_user = user  
        else:
            g.current_user = None


@app.context_processor
def inject_user():
    return dict(current_user=g.current_user)

@app.before_request
def setup():
    init_db()


def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

@app.route('/')
def home():
    logging.info(f"Current user in session: {g.current_user}")  
    if 'token' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        try:
            user = conn.execute(
                'SELECT * FROM users WHERE username = ?',
                (username,)
            ).fetchone()

            if user and check_password_hash(user['password'], password):  
                session['user_id'] = user['id'] 
                session['username'] = username
                session['token'] = generate_token(user['id'])
                flash('Login successful!', 'success')
                logging.info(f"User logged in: {username}")
                return redirect(url_for('home'))

            flash('Invalid credentials', 'error')
            logging.warning(f"Failed login attempt for user: {username}")

        except Exception as e:
            logging.error(f"Login error: {e}")
            flash('An error occurred', 'error')
        finally:
            conn.close()

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        hashed_password = generate_password_hash(password) 

        conn = get_db()
        try:
            conn.execute(
                'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                (username, hashed_password, email)  
            )
            conn.commit()
            flash('Registration successful!', 'success')
            logging.info(f"New user registered: {username}")
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
            logging.warning(f"Registration failed - duplicate user: {username}")
        except Exception as e:
            flash('An error occurred', 'error')
            logging.error(f"Registration error: {e}")
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/predict', methods=['GET', 'POST'])

def predict():
    if 'token' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        try:
            input_data = pd.DataFrame([[
                float(request.form['age']),
                float(request.form['experience']),
                float(request.form['income']),
                float(request.form['family']),
                float(request.form['cc_avg']),
                float(request.form['education']),
                float(request.form['mortgage']),
                float(request.form['securities_account']),
                float(request.form['cd_account']),
                float(request.form['online']),
                float(request.form['credit_card'])
            ]], columns=[
                "age", "experience", "income", "family", "cc_avg",
                "education", "mortgage", "securities_account", "cd_account",
                "online", "credit_card"
            ])
            feature_names = [
                "Age", "Experience", "Income", "Family", "CCAvg",
                "Education", "Mortgage", "Securities.Account", "CD.Account",
                "Online", "CreditCard"
            ]
            input_data.columns = feature_names

            model_choice = request.form['model_choice']
            if model_choice == 'model1':
                selected_model = model_1
                selected_scaler = scaler_1
                input_scaled = selected_scaler.transform(input_data)
                prediction = selected_model.predict(input_scaled)[0]


            elif model_choice == 'model2':
                selected_model = model_2
                scaler = StandardScaler()
                input_scaled = scaler.fit_transform(input_data)        
                X_test_tensor = torch.tensor(input_scaled, dtype=torch.float32)
                X_test_enc = crypten.cryptensor(X_test_tensor)
                y_pred_enc = selected_model(X_test_enc).get_plain_text()  
                prediction = (y_pred_enc > 0.5).float().numpy()
                prediction = int(prediction[0][0])
            else:
                flash('Invalid model selection', 'error')
                return redirect(url_for('predict'))


            user_id = verify_token(session['token'])

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (
                    user_id, age, experience, income, family, cc_avg, education, 
                    mortgage, securities_account, cd_account, online, credit_card, 
                    prediction, prediction_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (
                user_id,
                request.form['age'],
                request.form['experience'],
                request.form['income'],
                request.form['family'],
                request.form['cc_avg'],
                request.form['education'],
                request.form['mortgage'],
                request.form['securities_account'],
                request.form['cd_account'],
                request.form['online'],
                request.form['credit_card'],
                prediction
            ))

            conn.commit()
            conn.close()

            return render_template(
                'result.html',
                prediction=prediction,
                approval="Approved" if prediction == 1 else "Rejected"
            )

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            flash(f'Error making prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))

@app.route('/history')
def history():
    if 'token' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    try:
        user_id = verify_token(session['token'])

        cursor = conn.execute('''
            SELECT age, experience, income, family, cc_avg, education, mortgage,
                   securities_account, cd_account, online, credit_card, prediction, prediction_time
            FROM predictions 
            WHERE user_id = ? 
            ORDER BY prediction_time DESC
        ''', (user_id,))

        predictions = [
            dict(zip([column[0] for column in cursor.description], row))
            for row in cursor.fetchall()
        ]

        return render_template('history.html', predictions=predictions)
    
    except Exception as e:
        logging.error(f"Error fetching prediction history: {e}")
        flash('Error loading prediction history', 'error')
        return redirect(url_for('home'))
    
    finally:
        conn.close()
@app.route('/logout')   
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.makedirs('logs')
    app.run(debug=Config.DEBUG, port=PORT)





