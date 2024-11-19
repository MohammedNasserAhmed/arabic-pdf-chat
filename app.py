import gradio as gr
import os
import subprocess
import uuid
import fitz
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from gtts import gTTS
import sys
import pytesseract
from pdf2image import convert_from_path
from huggingface_hub import Repository, login
from huggingface_hub import hf_hub_download
from langchain.schema import Document
from PyPDF2 import PdfReader  
from langdetect import detect  


# Load environment variables
load_dotenv()
secret_key = os.getenv("GROQ_API_KEY")
hf_key = os.getenv("HF_TOKEN")

os.environ["GROQ_API_KEY"] = secret_key
login(token=hf_key,add_to_git_credential=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Ensure the necessary folders exist
UPLOAD_FOLDER = 'uploads/'
AUDIO_FOLDER = 'audio/'
for folder in [UPLOAD_FOLDER, AUDIO_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_pdf(file_path):
    """Load and preprocess Arabic text from a PDF file."""
    
    try:
        pages = convert_from_path(file_path, 500)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

    documents = []
    for pageNum, imgBlob in enumerate(pages):
        try:
            text = pytesseract.image_to_string(imgBlob, lang="ara")
            
            documents.append(text)
        except Exception as e:
            print(f"Error processing page {pageNum}: {e}")
            documents.append("")  

    return documents

def prepare_vectorstore(data):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separator="\n")
    # Create Document objects from the input data
    documents = [Document(page_content=text) for text in data]
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    # Create the vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(model="gemma2-9b-it", temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(llm=llm, output_key="answer", memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
        chain_type="map_reduce"
    )
    return chain


    
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

body {
    font-family: 'Noto Kufi Arabic', sans-serif;
    background: linear-gradient(135deg, #799351 0%, #A67B5B 100%);
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.gradio-container {
    direction: rtl;
    font-family: 'Noto Kufi Arabic', sans-serif;
    font-size: 16px;
    max-width: 800px !important;
    margin: auto !important;
    
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 20px;
}


.gr-textbox input, .gr-textbox textarea {
    text-align: right !important;  /* Align text to the right */
    direction: rtl !important;     /* Set RTL text direction */
    font-family: 'Cairo', sans-serif !important;
}



.gr-file, .gr-audio {
    text-align: right !important;  /* Align text to the right */
    direction: rtl !important;     /* Set RTL text direction */
}

label {
    font-size: 14px !important;
    color: #000000 !important;
    background-color: #EEEEEE;
}


.arabic-chatbox .message.user {
    font-family: 'Cairo', sans-serif !important;
    background-color: #FFFBE6; 
}

.arabic-chatbox .message.bot {
    font-family: 'Cairo', sans-serif !important;
    background-color: #E7FBE6; 
}

#custom-logo {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 30px;  /* Set custom width */
    height: 20px; /* Set custom height */
}

.custom-submit-button {
    background-color: #E68369 !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    cursor: pointer !important;
}

.custom-submit-button:hover {
    background-color: white !important;
    color: #E6B9A6 !important;
}

#clear_btn {
    background-color: #698474;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}

#clear_btn:hover {
    background-color: white;
    color: #698474;
}

"""

# Function to check if the file is a valid PDF in Arabic and less than 10MB
def validate_pdf(pdf):
    if pdf is None:
        return "لم يتم اختيار أي ملف", False
    if not pdf.name.endswith(".pdf"):
        return "الملف الذي اخترته ليس PDF", False
    if os.path.getsize(pdf.name) > 10 * 1024 * 1024:
        return "حجم الملف أكبر من 10 ميجا بايت", False
    
    # Check if PDF content is Arabic
    reader = PdfReader(pdf.name)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    try:
        if detect(text) != "ar":
            return "الملف ليس باللغة العربية", False
    except:
        return "فشل في تحليل اللغة", False
    
    return "الملف صالح للدردشة", True

def upload_pdf(pdf_file):
    global vectorstore, chathistory  
    chathistory = []
    data = load_pdf(pdf_file)
    vectorstore = prepare_vectorstore(data)
    
    return "تم تحميل الملف بنجاح !", True

        
def chat(user_input):
    global chathistory, vectorstore

    if not user_input.strip():  # Check if the input is empty or contains only whitespace
        return gr.update(value='<span style="color:red;">الرجاء إدخال سؤال.</span>'), "", None


   
    prompt = f"""
        You are an expert Arabic-language assistant specialized in analyzing and responding to queries about Arabic PDF documents. Your responses should be precise, informative, and reflect the professional tone and structure expected in formal Arabic communication. Focus on extracting and presenting relevant information from the document clearly and systematically, while avoiding colloquial or informal language.
        When responding, ensure the following:
           - Your answer directly reflects the content of the document.
           - If the requested information is not available in the document, clearly state that in Arabic.
           - Keep your response concise yet comprehensive, addressing the question fully.
           - Respond only in a professional and well-versed Arabic Language.
        Question: {user_input}.
        """
    chain = create_chain(vectorstore)
    response = chain({"question": prompt})
    assistant_response = response["answer"]

    

    chathistory.append({"user_content":  f"👤 {user_input}", "bot_content": f"🤖 {assistant_response}"})
    # Generate a unique identifier for the audio file
    audio_id = str(uuid.uuid4())

    # Create audio file
    tts = gTTS(text=assistant_response, lang='ar')
    audio_file = f"{audio_id}.mp3"
    tts.save(audio_file)

    history_display = [(msg["user_content"], msg["bot_content"]) for msg in chathistory]
    return gr.update(value=''), history_display, audio_file

image_path = "logo.png"
with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        gr.Image(image_path, show_fullscreen_button=False, show_download_button=False, 
                 show_share_button=False, show_label=False, label='', container=True, height=50, width=50)
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'>المساعد العربي للدردشة 💬</h1>", rtl=True)
    
    with gr.Row():
        gr.Markdown("""
                    
                    <ul style="list-style-type: disc;"> 
                        <li style="color: #6C946F; font-size: 12px;">تأكد من اختيار ملف PDF.</li>
                        <li style="color: #6C946F; font-size: 12px;">حجم الملف يجب أن يكون أقل من 10 ميجابايت.</li>
                        <li style="color: #6C946F; font-size: 12px;">يجب أن يكون المحتوى باللغة العربية.</li>
                    </ul>""", rtl=True)
        pdf_input = gr.File(label="اختر ملف PDF")
    with gr.Row():
        output_label = gr.HTML(value='')  
    with gr.Row():
        submit_button_pdf = gr.Button("ارفع الملف", interactive=False, variant='primary')
    with gr.Row():   
        chatbot = gr.Chatbot(label="الشات", height=400, rtl=True, show_copy_all_button=True, layout='bubble', scale=1, bubble_full_width=False)
    with gr.Row():
        chat_label = gr.HTML(value='')  
    with gr.Row():
        chat_input = gr.Textbox(label="💬", rtl=True, visible=False,  placeholder="أدخل سؤالك هنا ..", lines=2)
    with gr.Row():
        audio_output = gr.Audio(label="🔊", visible=False)
    
    with gr.Row():
        submit_button_chat = gr.Button("إرسال", interactive=True, visible=False, elem_classes="custom-submit-button", variant='primary')
    with gr.Row():
        clear_btn = gr.Button("مسح", interactive=True, visible=False, variant='secondary')

    def handle_file_upload(pdf):
        output_label.value=''
        message, is_valid = validate_pdf(pdf)
        color = "red" if not is_valid else "green"
        # Update HTML label instead of Textbox
        
        if is_valid:
            # Enable the upload button if the file is valid
            value=''
            return gr.update(value=value), gr.update(interactive=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
        else:
            value=f'<span style="color:{color}">{message}</span>'
            return gr.update(value=value), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def process_pdf_and_enable_components(pdf):
        # Process PDF and activate the other components
        output_label.value='<span style="color:blue">جاري معالجة الملف...</span>'
        message, is_valid = upload_pdf(pdf)
        value=f'<span style="color:green">{message}</span>'
        return gr.update(value=value), gr.update(visible=True), gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    def clear_chat():
        return "", None 
       
     # When the user uploads a file, validate it and then allow PDF upload
    pdf_input.change(handle_file_upload, inputs=pdf_input, outputs=[output_label,submit_button_pdf, submit_button_chat, chatbot, chat_input, audio_output, clear_btn])

    # When the user presses the upload button, process the PDF and enable other components
    submit_button_pdf.click(process_pdf_and_enable_components, inputs=pdf_input, outputs=[output_label, submit_button_chat, submit_button_pdf, chatbot, chat_input, audio_output, clear_btn])
    clear_btn.click(clear_chat, outputs=[chat_input, audio_output])
    # Chat button connection
    submit_button_chat.click(chat, inputs=chat_input, outputs=[chat_label, chatbot, audio_output])
    

# Launch the Gradio app
demo.launch(inbrowser=True)



