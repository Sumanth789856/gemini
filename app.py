from flask import Flask, render_template, request, redirect, url_for, session, flash
import pymysql
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import request, render_template
import google.generativeai as genai
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database Connection
db = pymysql.connect(
    host="localhost",
    user="root",
    password="7842909856a@A",
    database="crop_db1"
)
cursor = db.cursor()

# Load and train the ML model if not already trained
try:
    with open('models/crop_model.pkl', 'rb') as file:
        crop_model = pickle.load(file)
except FileNotFoundError:
    # Train model if it doesn't exist
    df = pd.read_csv('models/crop_data.csv')
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    crop_model = RandomForestClassifier()
    crop_model.fit(X_train, y_train)
    with open('models/crop_model.pkl', 'wb') as file:
        pickle.dump(crop_model, file)

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('predict'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        # Store user details in MySQL database
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        db.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Verify user credentials
        cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cursor.fetchone()
        if user:
            session['user_id'] = user[0]  # Save user ID in session
            return redirect(url_for('predict'))
        flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' in session:
        if request.method == 'POST':
            n = float(request.form['n'])
            p = float(request.form['p'])
            k = float(request.form['k'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Predict the crop
            features = [[n, p, k, temperature, humidity, ph, rainfall]]
            predicted_crop = crop_model.predict(features)[0]

            # Crop dictionary
            crop_images = {
                'rice': 'crop_images/rice.png',
                'wheat': 'crop_images/wheat.png',
                'maize': 'crop_images/maize.jpeg',
                'apple': 'crop_images/apple.jpg',
                'banana': 'crop_images/banana.jpeg',
                'chickpea': 'crop_images/chickpea.jpg',
                'coconut': 'crop_images/coconut.jpg',
                'coffee': 'crop_images/coffee.jpg',
                'cotton': 'crop_images/cotton.jpeg',
                'grapes': 'crop_images/grapes.jpeg',
                'jute': 'crop_images/jute.jpeg',
                'kidneybeans': 'crop_images/kidneybeans.jpeg',
                'lentil': 'crop_images/lentil.jpeg',
                'maize': 'crop_images/maize.jpeg',
                'mango': 'crop_images/mango.jpeg',
                'mothbeans': 'crop_images/mothbeans.jpeg',
                'mungbeans': 'crop_images/mungbeans.jpeg',
                'muskmelon': 'crop_images/muskmelon.jpeg',
                'orange': 'crop_images/orange.jpeg',
                'papaya': 'crop_images/papaya.jpeg',
                'pigeonpeas': 'crop_images/pigeonpeas.jpeg',
                'pomergranate': 'crop_images/pomergranate.jpg',
                'watermelon': 'crop_images/watermelon.jpg'
                # Add more crops and their image paths here
            }

            # Check if the predicted crop exists in the dictionary
            crop_image = crop_images.get(predicted_crop, 'crop_images/default.jpg')  # Default image if not found

            return render_template('result1.html', crop=predicted_crop, crop_image=crop_image)
        return render_template('predict.html')
    return redirect(url_for('login'))
@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Store user details in the database
        cursor.execute("INSERT INTO contact1 (name, email, message) VALUES (%s, %s, %s)", (name, email, message))
        db.commit()
        flash('Thank you for reaching out! We will get back to you soon.')
        return redirect(url_for('contactus'))
    return render_template('contactus.html')
# Configure Gemini


genai.configure(api_key='AIzaSyAitsyh3co7B5D4D0NmcUBhD0TsKM6qbPs')  # Key hardcoded
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Secure file handling
            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            upload_dir = os.path.join('static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            img_path = os.path.join(upload_dir, filename)
            file.save(img_path)

            try:
                # Generate analysis with Gemini
                response = model.generate_content([
                    "Analyze this crop image and provide the following in MARKDOWN format:\n"
                    "**Crop Name:** <crop>\n"
                    "**Disease:** <disease>\n"
                    "**Recommendations:**\n"
                    "- <list of pesticides>\n"
                    "- <prevention methods>\n\n"
                    "Include emojis where appropriate 🌱🌾",
                    genai.upload_file(img_path)
                ])

                # Parse response
                result_data = {
                    'crop': 'Unknown Crop',
                    'disease': 'No Disease Detected',
                    'recommendations': ['No recommendations available'],
                    'image': img_path,
                    'status_icon': '⚠️'
                }

                if response.text:
                    content = response.text.split('\n')
                    for line in content:
                        if '**Crop Name:**' in line:
                            result_data['crop'] = line.split('**Crop Name:**')[-1].strip()
                        elif '**Disease:**' in line:
                            result_data['disease'] = line.split('**Disease:**')[-1].strip()
                            result_data['status_icon'] = '✅' if 'No Disease' in result_data['disease'] else '⚠️'
                        elif '**Recommendations:**' in line:
                            recommendations = []
                            for item in content[content.index(line)+1:]:
                                if item.strip().startswith('-'):
                                    recommendations.append(item.strip()[1:].strip())
                            result_data['recommendations'] = recommendations
                            break

                return render_template(
                    'interactive_result.html',
                    **result_data,
                    original_filename=filename
                )

            except Exception as e:
                return render_template(
                    'error.html',
                    error_message=f"Analysis failed: {str(e)}",
                    recovery_tip="Try uploading a clearer image of the crop leaves"
                )

    # GET request or failed POST
    return render_template('detect.html')
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
