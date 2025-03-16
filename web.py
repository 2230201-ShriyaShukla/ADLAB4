from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xls', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the upload page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload
@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'document' not in request.files:
        return "No file part", 400
    file = request.files['document']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('chatbot'))  # Redirect to the chatbot page
    return "Invalid file type", 400

# Route for the chatbot page
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# Route to handle chatbot queries
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')
    model_choice = data.get('model_choice')
    
    # Process the query and generate a response (dummy response for now)
    answer = f"This is a response from {model_choice} for the query: {query}"
    
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)