import os
from flask import Flask, request, jsonify
from google.cloud import vision
import json
import firebase_admin
from firebase_admin import credentials, auth
import docx
import io
from flask_cors import CORS
import requests

# --- CONFIGURATION (NOW WITH DETAILED LOGGING) ---
print("--- SERVER STARTING: CONFIGURATION PHASE ---")

# --- 1. FIREBASE ADMIN CREDENTIALS SETUP ---
try:
    print("Attempting to load Firebase credentials...")
    firebase_creds_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if firebase_creds_json_str:
        print("SUCCESS: Found FIREBASE_CREDENTIALS_JSON environment variable.")
        firebase_creds_dict = json.loads(firebase_creds_json_str)
        cred = credentials.Certificate(firebase_creds_dict)
    else:
        print("INFO: FIREBASE_CREDENTIALS_JSON env var not found. Trying local file 'firebase-admin-credentials.json'...")
        cred = credentials.Certificate("firebase-admin-credentials.json")
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    print("SUCCESS: Firebase Admin initialized.")
except Exception as e:
    print(f"!!! FATAL ERROR INITIALIZING FIREBASE ADMIN: {e}")

# --- 2. GOOGLE CLOUD VISION CREDENTIALS SETUP ---
try:
    print("Attempting to load Google Cloud Vision credentials...")
    google_creds_json_str = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if google_creds_json_str:
        print("SUCCESS: Found GOOGLE_CREDENTIALS_JSON environment variable.")
        creds_path = "/tmp/credentials.json"
        with open(creds_path, "w") as f:
            f.write(google_creds_json_str)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
        print(f"SUCCESS: Wrote Google credentials to {creds_path}")
    else:
        print("INFO: GOOGLE_CREDENTIALS_JSON env var not found. Using local file 'credentials.json'...")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
except Exception as e:
    print(f"!!! FATAL ERROR SETTING UP GOOGLE CLOUD VISION CREDS: {e}")

# --- 3. API KEY AND PROJECT ID SETUP ---
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

if not API_KEY:
    print("!!! WARNING: API_KEY environment variable not found! Using fallback for local dev.")
    API_KEY = "AIzaSyCmmjmRZdhVfRhTh2NN9AdcnspKLKqaVlc"

if not PROJECT_ID:
    print("!!! WARNING: PROJECT_ID environment variable not found! Using fallback for local dev.")
    PROJECT_ID = "helical-realm-472708-n9"

print(f"--- Configuration complete. Project ID: {PROJECT_ID} ---")

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- All other functions and routes remain unchanged ---
def get_text_from_file(file_content, mime_type):
    if not file_content: raise Exception("The uploaded file is empty.")
    if 'wordprocessingml.document' in mime_type:
        try:
            doc = docx.Document(io.BytesIO(file_content))
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e: raise Exception(f"Error reading .docx file: {e}")
    if mime_type.startswith(('application/pdf', 'image/')):
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=file_content)
        response = client.document_text_detection(image=image)
        if response.error.message: raise Exception(f"Vision AI Error: {response.error.message}")
        return response.full_text_annotation.text
    if mime_type.startswith('text/'):
        try: return file_content.decode('utf-8')
        except UnicodeDecodeError: raise Exception("Could not read text file.")
    raise Exception(f"Unsupported file type: {mime_type}.")

def call_gemini_api(prompt):
    model_name = "gemini-1.5-flash-latest"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Google API Error: {response.status_code} {response.text}")
    response_json = response.json()
    if not response_json.get('candidates'):
        raise Exception("API returned no candidates. The prompt may have been blocked.")
    return response_json['candidates'][0]['content']['parts'][0]['text']

def analyze_document_with_gemini(document_text):
    prompt = f'Analyze the following legal document and provide a structured JSON analysis with keys "fairnessScore", "summary", and "riskRadar". The riskRadar should be an array of objects, each with "level", "title", and "explanation".\n\nDOCUMENT:\n{document_text}'
    response_text = call_gemini_api(prompt)
    cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
    try:
        analysis_data = json.loads(cleaned_text)
        return analysis_data
    except json.JSONDecodeError:
        return {"fairnessScore": 0, "summary": "Could not analyze.", "riskRadar": []}

def reformat_document_with_gemini(document_text):
    prompt = f"Reformat the following text for maximum readability, applying proper paragraphs and structure. Do not change the wording. Return only the formatted text.\n\nDOCUMENT:\n{document_text}"
    return call_gemini_api(prompt)

def simplify_document_with_gemini(document_text):
    prompt = f"Rewrite the following document in plain, easy-to-understand language. Maintain the original structure. Return only the simplified text.\n\nDOCUMENT:\n{document_text}"
    return call_gemini_api(prompt)

def answer_chat_question(document_text, question):
    prompt = f"Based ONLY on the document text provided, answer the user's question concisely.\n\nDOCUMENT:\n{document_text}\n\nQUESTION:\n{question}"
    return call_gemini_api(prompt)

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '': return jsonify({"error": "No file selected"}), 400
        raw_text = get_text_from_file(file.read(), file.mimetype)
        formatted_text = reformat_document_with_gemini(raw_text)
        analysis_data = analyze_document_with_gemini(formatted_text)
        return jsonify({"analysis": analysis_data, "fullText": formatted_text}), 200
    except Exception as e:
        print(f"!!! ERROR in /analyze: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/simplify', methods=['POST'])
def simplify_endpoint():
    try:
        data = request.get_json()
        if not data or 'documentText' not in data: return jsonify({"error": "Missing document text"}), 400
        simplified_text = simplify_document_with_gemini(data['documentText'])
        return jsonify({"simplifiedText": simplified_text})
    except Exception as e:
        print(f"!!! ERROR in /simplify: {e}")
        return jsonify({"error": "Could not simplify the document."}), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'documentText' not in data:
            return jsonify({"error": "Missing data"}), 400
        answer = answer_chat_question(data['documentText'], data['question'])
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"!!! ERROR in /chat: {e}")
        return jsonify({"error": "Could not process question."}), 500

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password: 
            return jsonify({'error': 'Email and password are required'}), 400
        user = auth..create_user(email=email, password=password)
        return jsonify({'uid': user.uid}), 201
    except Exception as e:
        print(f"!!! ERROR in /signup: {e}")
        error_message = "An unknown error occurred."
        if hasattr(e, 'code'):
            if e.code == 'EMAIL_EXISTS': error_message = 'The email address is already in use.'
            elif e.code == 'WEAK_PASSWORD': error_message = 'Password should be at least 6 characters.'
        return jsonify({'error': error_message}), 400

@app.route('/')
def index(): return app.send_static_file('index.html')
@app.route('/login.html')
def login_page(): return app.send_static_file('login.html')
@app.route('/dashboard.html')
def dashboard(): return app.send_static_file('dashboard.html')
@app.route('/privacy.html')
def privacy(): return app.send_static_file('privacy.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

