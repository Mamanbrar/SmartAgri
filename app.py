from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
from PIL import Image
from torchvision import transforms
import timm
import pandas as pd
import sqlite3
from flask import flash , session
import pickle
# Initialize the Flask app
app = Flask(__name__)

# Route for the Home page (index)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/thank')
def thank():
    return render_template('thankyou.html')

# Load the model
num_classes = 17  # Match this with your number of classes
model = timm.create_model("rexnet_150", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('crop_best_model.pth', map_location=torch.device('cpu'), weights_only=True))

model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names dictionary
class_names = {
    0: 'Corn___Common_Rust', 1: 'Corn___Gray_Leaf_Spot', 2: 'Corn___Healthy', 3: 'Corn___Northern_Leaf_Blight',
    4: 'Potato___Early_Blight', 5: 'Potato___Healthy', 6: 'Potato___Late_Blight', 7: 'Rice___Brown_Spot',
    8: 'Rice___Healthy', 9: 'Rice___Leaf_Blast', 10: 'Rice___Neck_Blast', 11: 'Sugarcane_Bacterial Blight',
    12: 'Sugarcane_Healthy', 13: 'Sugarcane_Red Rot', 14: 'Wheat___Brown_Rust', 15: 'Wheat___Healthy', 16: 'Wheat___Yellow_Rust'
}
#efine routes
@app.route("/Disease", methods=["GET", "POST"])
def Disease():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = f"static/{file.filename}"
            file.save(file_path)
            disease_name = predict_disease(file_path)
            return render_template("Disease.html", disease_name=disease_name, image_path=file_path)
    return render_template("Disease.html", disease_name=None)

# Prediction function
def predict_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    return class_names.get(predicted_class.item(), "Unknown")


# Load crop data from Excel
def load_crop_data():
    df = pd.read_excel("crop_calendar.xlsx")
    crop_data = df.set_index('Crop').T.to_dict()
    return crop_data

# Load data once when the app starts
crop_data = load_crop_data()

@app.route('/calender')
def calender():
    # Pass crop names to the template for the dropdown
    crop_names = list(crop_data.keys())
    return render_template('calender.html', crop_names=crop_names)

@app.route('/get_crop_info', methods=['POST'])
def get_crop_info():
    crop_name = request.form['crop_name']
    # Fetch the crop details
    crop_info = crop_data.get(crop_name, {})
    return jsonify(crop_info)


#about form
def init_db():
    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contact_form (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL,
                message TEXT NOT NULL
            );
        ''')
        conn.commit()

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    if request.method == 'POST':
        full_name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Insert data into the database
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO contact_form (full_name, email, message) VALUES (?, ?, ?)
            ''', (full_name, email, message))
            conn.commit()

        return render_template('thankyou.html', full_name=full_name)  # Redirect to a thank you page


#signup page/signin 

app.secret_key = 'kamalveer28'
def get_db_connection():
    conn = sqlite3.connect('users.db')  # The SQLite database file
    conn.row_factory = sqlite3.Row  # This allows us to treat rows as dictionaries
    return conn


# Create the users table if it doesn't exist
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()


# SQLite3 database connection

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        
        if password != confirm_password:
            flash('Passwords do not match! Please try again.', 'danger')
            return render_template('signup.html')
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Insert the new user's data into the database
            cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
            conn.commit()

            cursor.close()
            conn.close()

            # Redirect to the Sign-In page after successful sign-up
            flash('Account created successfully! Please sign in.', 'success')
            return redirect(url_for('signin'))

        except sqlite3.IntegrityError:
            flash('Error: This email is already registered.', 'danger')
            return render_template('signup.html')

    return render_template('signup.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if the email exists in the database
        cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user:
            flash('Sign-In Successful!', 'success')
            return redirect(url_for('dashboard'))  # Replace with your dashboard or home route
        else:
            flash('Invalid email or password. Please try again.', 'danger')
            return redirect(url_for('signin'))

    return render_template('signin.html')


@app.route('/dashboard')
def dashboard():
    return "successful"  # This is just a placeholder for the dashboard


#agriculture optimization
loadedModel = pickle.load(open('agriculture.pkl', 'rb')) 

@app.route('/opti')
def opti():
    return render_template('opti.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorous'])
    K = int(request.form['Pottasium'])
    temperature = int(request.form['Temperature'])
    humidity = int(request.form['Humidity'])
    ph = int(request.form['PH'])
    rainfall = int(request.form['Rainfall'])
    
    Suitable_Crop = loadedModel.predict([[N,P,K,temperature,humidity,ph,rainfall]])[0]

    return render_template('result.html', output=Suitable_Crop)




if __name__ == '__main__':
    init_db()  # Create the database and table if it doesn't exist
    app.run()
