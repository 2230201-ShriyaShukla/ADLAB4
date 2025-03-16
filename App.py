import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# For document processing
import pandas as pd

# For LangChain integration (later for querying)
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# For the example LLM (using HuggingFaceHub)
from langchain.llms import HuggingFaceHub

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it does not exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable to store our document index (for simplicity)
vector_index = None

###########################
# Helper Functions
###########################

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file, with a fallback method."""
    from PyPDF2 import PdfReader
    from pdfminer.high_level import extract_text  # Backup method

    text = ""
    try:
        print(f"📂 Extracting text from: {file_path}")

        # First attempt: PyPDF2
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # If PyPDF2 fails, use pdfminer.six
        if not text.strip():
            print("⚠️ PyPDF2 failed, switching to pdfminer.six")
            text = extract_text(file_path)

        if text.strip():
            print(f"✅ Extracted Text (first 500 chars): {text[:500]}")
        else:
            print("⚠️ No readable text found. The PDF may contain only images.")

    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return "Error extracting text."

    return text if text.strip() else "No readable text found in PDF."

def extract_text_from_docx(file_path):
    """Extract text from a Word document."""
    import docx
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_excel(file_path):
    """Extract text from an Excel file."""
    df = pd.read_excel(file_path)
    return df.to_string()

def create_vector_index(text):
    """Split the text into chunks and create a vector index using FAISS."""
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_texts(texts, embeddings)
    return index

###########################
# Routes
###########################

@app.route('/')
def index():
    """Render the document upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload:
      - Save the uploaded file.
      - Extract text based on file type.
      - Create an index (vector store) from the text.
      - Then redirect (or render) the chat interface.
    """
    global vector_index
    if 'document' not in request.files:
        return "No file part", 400

    file = request.files['document']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    print(f"✅ File '{filename}' uploaded successfully!")

    # Determine file type and extract text
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif filename.lower().endswith(('.docx', '.doc')):
        text = extract_text_from_docx(file_path)
    elif filename.lower().endswith(('.xls', '.xlsx')):
        text = extract_text_from_excel(file_path)
    else:
        return "Unsupported file format", 400

    # Create a vector index from the extracted text
    if text.strip():
        vector_index = create_vector_index(text)
        print("✅ Vector index created successfully!")
    else:
        return "No readable text found in document.", 400

    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    """
    Receive a user query and model choice from the frontend,
    retrieve relevant document chunks using the vector index,
    and then use an LLM to generate an answer.
    """
    global vector_index
    if vector_index is None:
        return jsonify({'answer': 'Please upload a document first.'})
    
    data = request.get_json()
    query = data.get('query', '')
    model_choice = data.get('model_choice', 'default')

    # Retrieve top relevant chunks from the vector index
    docs = vector_index.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])

    # Create a prompt that includes the retrieved context
    prompt = f"Given the context:\n{context}\nAnswer the question:\n{query}"

    # Select LLM Model
    if model_choice == 'LLama':
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":1e-10})
    elif model_choice == 'Gemma':
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":1e-10})
    else:
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":1e-10})

    answer = llm(prompt)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)

