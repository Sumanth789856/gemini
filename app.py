from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
import pymysql
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import request, render_template
import google.generativeai as genai
import os
import joblib
from werkzeug.utils import secure_filename 
import requests
from datetime import datetime, timedelta 




app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database Connection
db = pymysql.connect(
    host="localhost",
    user="root",
    password="7842909856a@A",
    database="crop_db12"
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
def landing():
    return render_template('landing.html')



def get_weather_data(location):
    WEATHERAPI_KEY = os.getenv('WEATHERAPI_KEY', 'b3dcd58dc02e4e58bb955159250204')
    try:
        # Construct API URL based on input type (city name or coordinates)
        if isinstance(location, dict) and 'lat' in location:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={location['lat']},{location['lon']}&days=7&aqi=no&alerts=no"
        else:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={location}&days=7&aqi=no&alerts=no"
        
        response = requests.get(url)
        data = response.json()
        
        if 'error' in data:
            return {'error': data['error']['message']}
        
        # Process the data
        processed_data = {
            'location': f"{data['location']['name']}, {data['location']['country']}",
            'current': {
                'temp_c': data['current']['temp_c'],
                'feelslike_c': data['current']['feelslike_c'],
                'condition': data['current']['condition']['text'],
                'icon': data['current']['condition']['icon'],
                'humidity': data['current']['humidity'],
                'rainfall': data['current']['precip_mm'],
                'wind_kph': data['current']['wind_kph'],
            },
            'hourly': [],
            'daily': []
        }
        
        # Process hourly data (next 24 hours)
        for hour in data['forecast']['forecastday'][0]['hour']:
            time = datetime.strptime(hour['time'], '%Y-%m-%d %H:%M').strftime('%H:%M')
            processed_data['hourly'].append({
                'time': time,
                'temp_c': hour['temp_c'],
                'humidity': hour['humidity'],
                'rainfall': hour['precip_mm'],
                'condition': hour['condition']['text'],
                'icon': hour['condition']['icon'],
                'chance_of_rain': hour['chance_of_rain']
            })
        
        # Process daily forecast
        for day in data['forecast']['forecastday']:
            date = datetime.strptime(day['date'], '%Y-%m-%d').strftime('%m/%d')
            weekday = datetime.strptime(day['date'], '%Y-%m-%d').strftime('%a')
            processed_data['daily'].append({
                'date': date,
                'day': weekday,
                'high': day['day']['maxtemp_c'],
                'low': day['day']['mintemp_c'],
                'humidity': day['day']['avghumidity'],
                'total_rainfall': day['day']['totalprecip_mm'],
                'condition': day['day']['condition']['text'],
                'icon': day['day']['condition']['icon'],
                'chance_of_rain': day['day']['daily_chance_of_rain']
            })
        
        return processed_data
        
    except Exception as e:
        return {'error': str(e)}
@app.route('/get_weather', methods=['POST'])
def get_weather():
    data = request.get_json()
    if 'city' in data:
        weather_data = get_weather_data(data['city'])
    elif 'lat' in data and 'lon' in data:
        weather_data = get_weather_data({'lat': data['lat'], 'lon': data['lon']})
    else:
        weather_data = {'error': 'Invalid request parameters'}
    return jsonify(weather_data)

@app.route('/weather')
def weather():
    if 'user_id' in session:
        return render_template('weather.html')

@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html')
    return render_template('login.html')

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to Check Allowed File Extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        # Handle File Upload
        if 'profile_pic' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['profile_pic']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Store user details and profile picture filename in MySQL
            cursor.execute("INSERT INTO users (name, email, password, profile_pic) VALUES (%s, %s, %s, %s)", 
                           (name, email, password, filename))
            db.commit()

            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        
        flash('Invalid file type. Allowed types: png, jpg, jpeg, gif')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor.execute("SELECT id, name, profile_pic FROM users WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()

        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['profile_pic'] = user[2] if user[2] else 'default_profile.png'  # Store in session
            return redirect(url_for('weather'))
        else:
            flash("Invalid email or password")

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
                'barley': 'crop_images/Barley.jpg',
                'banana': 'crop_images/banana.jpeg',
                'chickpea': 'crop_images/chickpea.jpg',
                'coconut': 'crop_images/coconut.jpg',
                'coffee': 'crop_images/coffee.jpg',
                'cotton': 'crop_images/cotton.jpeg',
                'grapes': 'crop_images/grapes.jpeg',
                'sugarcane': 'crop_images/Sugarcane.png',
                'chili': 'crop_images/Chili.png',
                'potato': 'crop_images/Potato.png',
                'tomato': 'crop_images/Tomato.png',
                'soybean': 'crop_images/Soybean.png',
                'kidneybeans': 'crop_images/kidneybeans.jpeg',
                'lentil': 'crop_images/lentil.jpeg',
                'mango': 'crop_images/mango.jpeg',
                'mothbeans': 'crop_images/mothbeans.jpeg',
                'mungbeans': 'crop_images/mungbeans.jpeg',
                'muskmelon': 'crop_images/muskmelon.jpeg',
                'orange': 'crop_images/orange.jpeg',
                'papaya': 'crop_images/papaya.jpeg',
                'pigeonpeas': 'crop_images/pigeonpeas.jpeg',
                'pomegranate': 'crop_images/pomergranate.jpg',
                'watermelon': 'crop_images/watermelon.jpg'
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




@app.route('/detect', methods=['GET', 'POST'])
def detect():
    genai.configure(api_key='AIzaSyD0GWPhKt5sQk957ASwiNYz3BP-a4gLsXU')  # Key hardcoded
    model = genai.GenerativeModel('gemini-1.5-flash')
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
    return redirect(url_for('landing'))
    


@app.route('/rotation', methods=['GET', 'POST'])
def rotation():
    model = joblib.load("crop_rotation_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    if 'user_id' in session:
        if request.method == 'POST':
            try:
                # Get user inputs
                crop_name = request.form['crop_name']
                n = float(request.form['n'])
                p = float(request.form['p'])
                k = float(request.form['k'])
                soil_type = request.form['soil_type']
                season = request.form['season']
                temperature = float(request.form['temperature'])
                rainfall = float(request.form['rainfall'])
                humidity = float(request.form['humidity'])
                
                # Encode categorical inputs
                encoded_crop_name = label_encoders['Crop Name'].transform([crop_name])[0]
                encoded_soil_type = label_encoders['Soil Type'].transform([soil_type])[0]
                encoded_season = label_encoders['Season'].transform([season])[0]

                # Prepare input for prediction
                input_data = np.array([[encoded_crop_name, n, p, k, encoded_soil_type, encoded_season, temperature, rainfall, humidity]])

                # Predict next crop
                predicted_crop_encoded = model.predict(input_data)[0]
                predicted_next_crop = label_encoders['Preferred Next Crop'].inverse_transform([predicted_crop_encoded])[0]
                
                # Convert predicted crop name to lowercase for consistent matching
                predicted_next_crop_lower = predicted_next_crop.lower()
                
                # Create a dictionary mapping crop names to their image paths with all possible variations
                crop_images = {
                    'rice': 'crop_images/Rice.png',
                    'wheat': 'crop_images/wheat.png',
                    'maize': 'crop_images/maize.jpeg',
                    'corn': 'crop_images/maize.jpeg',  # alternative name for maize
                    'apple': 'crop_images/apple.jpg',
                    'barley': 'crop_images/Barley.jpg',
                    'banana': 'crop_images/banana.jpeg',
                    'blackgram': 'crop_images/blackgram.jpg',
                    'chickpea': 'crop_images/chickpea.jpg',
                    'chick pea': 'crop_images/chickpea.jpg',
                    'chili': 'crop_images/Chili.png',
                    'coconut': 'crop_images/coconut.jpg',
                    'coffee': 'crop_images/coffee.jpg',
                    'cotton': 'crop_images/cotton.jpeg',
                    'grapes': 'crop_images/grapes.jpeg',
                    'jute': 'crop_images/jute.jpeg',
                    'kidneybeans': 'crop_images/kidneybeans.jpeg',
                    'kidney beans': 'crop_images/kidneybeans.jpeg',
                    'lentil': 'crop_images/lentil.jpeg',
                    'lentils': 'crop_images/lentil.jpeg',
                    'mango': 'crop_images/mango.jpeg',
                    'mothbeans': 'crop_images/mothbeans.jpeg',
                    'moth beans': 'crop_images/mothbeans.jpeg',
                    'mungbeans': 'crop_images/mungbeans.jpeg',
                    'mung beans': 'crop_images/mungbeans.jpeg',
                    'muskmelon': 'crop_images/muskmelon.jpeg',
                    'orange': 'crop_images/orange.jpeg',
                    'papaya': 'crop_images/papaya.jpeg',
                    'pigeonpeas': 'crop_images/pigeonpeas.jpeg',
                    'pigeon peas': 'crop_images/pigeonpeas.jpeg',
                    'pomegranate': 'crop_images/pomergranate.jpg',
                    'tomato': 'crop_images/Tomato.png',
                    'watermelon': 'crop_images/watermelon.jpg'
                }
                
                # First try exact match, then try lowercase match
                crop_image = crop_images.get(predicted_next_crop) or \
                            crop_images.get(predicted_next_crop_lower) or \
                            crop_images.get(predicted_next_crop_lower.replace(' ', '')) or \
                            'crop_images/default.jpg'

                # Verify if the image file exists in the static folder
                image_path = os.path.join('static', crop_image)
                if not os.path.exists(image_path):
                    crop_image = 'crop_images/default.jpg'

                return render_template('rotation_result.html', 
                                     next_crop=predicted_next_crop,
                                     crop_image=crop_image)

            except Exception as e:
                return render_template('rotation.html', error="❌ Error: Invalid input values or unrecognized crop/season/soil type.")

        return render_template('rotation.html')
@app.route('/create_post', methods=['POST'])
def create_post():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    content = request.form.get('content')
    image = request.files.get('image')

    image_filename = None
    if image and allowed_file(image.filename):
        image_filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path)

    # Insert into MySQL
    cursor.execute("INSERT INTO posts (user_id, content, image) VALUES (%s, %s, %s)", 
                   (user_id, content, image_filename))
    db.commit()

    return redirect(url_for('community'))
@app.route('/community')
def community():
    cursor.execute("""
        SELECT posts.id, users.name, posts.content, posts.image, posts.created_at, 
               COALESCE(users.profile_pic, 'default_profile.png') AS profile_picture,
               (SELECT COUNT(*) FROM likes WHERE likes.post_id = posts.id) AS like_count
        FROM posts 
        JOIN users ON posts.user_id = users.id 
        ORDER BY posts.created_at DESC
    """)
    posts = cursor.fetchall()

    # Fetch comments for each post
    comments = {}
    for post in posts:
        cursor.execute("""
            SELECT comments.comment, users.name, users.profile_pic 
            FROM comments 
            JOIN users ON comments.user_id = users.id 
            WHERE comments.post_id = %s
            ORDER BY comments.created_at ASC
        """, (post[0],))
        comments[post[0]] = cursor.fetchall()

    return render_template('community.html', posts=posts, comments=comments)

@app.route('/like/<int:post_id>', methods=['POST'])
def like_post(post_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    
    # Check if the user already liked the post
    cursor.execute("SELECT * FROM likes WHERE user_id = %s AND post_id = %s", (user_id, post_id))
    existing_like = cursor.fetchone()

    if not existing_like:
        cursor.execute("INSERT INTO likes (user_id, post_id) VALUES (%s, %s)", (user_id, post_id))
        db.commit()

    return redirect(url_for('community'))
@app.route('/comment/<int:post_id>', methods=['POST'])
def comment_post(post_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    comment_text = request.form['comment']

    if comment_text.strip():
        cursor.execute("INSERT INTO comments (user_id, post_id, comment) VALUES (%s, %s, %s)", 
                       (user_id, post_id, comment_text))
        db.commit()

    return redirect(url_for('community'))
import os

@app.route('/delete_post/<int:post_id>', methods=['POST'])
def delete_post(post_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Fetch the image filename before deleting the post
    cursor.execute("SELECT content FROM posts WHERE id = %s", (post_id,))
    post = cursor.fetchone()

    if post and post[0]:  # If the post exists and has an image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], post[0])
        if os.path.exists(image_path):  # Check if the file exists
            os.remove(image_path)  # Delete the file

    # Delete the post from the database
    cursor.execute("DELETE FROM posts WHERE id = %s", (post_id,))
   # cursor.execute("DELETE FROM comments WHERE post_id = %s", (post_id,))
    db.commit()

    return redirect(url_for('community'))


@app.route('/my_posts')
def my_posts():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']

    # Fetch the logged-in user's posts
    cursor.execute("""
        SELECT posts.id, users.name, posts.content, posts.image, posts.created_at, 
               COALESCE(users.profile_pic, 'default_profile.png') AS profile_picture,
               (SELECT COUNT(*) FROM likes WHERE likes.post_id = posts.id) AS like_count
        FROM posts 
        JOIN users ON posts.user_id = users.id 
        WHERE posts.user_id = %s
        ORDER BY posts.created_at DESC
    """, (user_id,))
    
    user_posts = cursor.fetchall()

    # Fetch comments for each post
    user_comments = {}
    for post in user_posts:
        cursor.execute("""
            SELECT comments.comment, users.name, users.profile_pic 
            FROM comments 
            JOIN users ON comments.user_id = users.id 
            WHERE comments.post_id = %s
            ORDER BY comments.created_at ASC
        """, (post[0],))
        user_comments[post[0]] = cursor.fetchall()

    return render_template('my_posts.html', posts=user_posts, comments=user_comments)




    

if __name__ == '__main__':
    app.run(debug=True,port=5001)
