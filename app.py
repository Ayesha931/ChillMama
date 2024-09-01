import os
import speech_recognition as sr
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from groq import Groq
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()

# Initialize Groq API client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize model and tokenizer for embedding
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Initialize vector database
dimension = 768  # Size of BERT embeddings
index = faiss.IndexFlatL2(dimension)

# Folder path containing PDFs
pdf_folder_path = "pdfsforRAG"

# Function to convert audio file to text
def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        print(f"Extracted Text: {text}")  # Debugging line
        return text
    except sr.UnknownValueError:
        print("Audio could not be understood")  # Debugging line
        return None
    except sr.RequestError:
        print("Request error")  # Debugging line
        return None

# Function to convert audio to WAV format
def convert_to_wav(audio_file_path):
    if not audio_file_path:
        raise ValueError("Invalid audio file path")
    try:
        audio = AudioSegment.from_file(audio_file_path)
        wav_path = "temp_audio.wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error converting audio to WAV: {e}")
        return None

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_document = fitz.open(pdf_file)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Function to embed text using a transformer model
def embed_text(texts, model, tokenizer):
    try:
        inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
        return embeddings
    except Exception as e:
        print(f"Error embedding text: {e}")
        return np.array([])  # Return empty array on error

# Function to convert text to speech
def text_to_speech(text, output_file):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        return output_file
    except Exception as e:
        print(f"Error converting text to speech: {e}")
        return None

# Read all PDF files from the specified folder
pdf_paths = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

texts = []
for path in pdf_paths:
    pdf_text = extract_text_from_pdf(path)
    if pdf_text:
        texts.append(pdf_text)
    else:
        print(f"Failed to extract text from {path}")

# Embed PDF texts and add to vector database
embeddings = embed_text(texts, model, tokenizer)
if embeddings.size > 0:
    index.add(embeddings)
else:
    print("No embeddings to add to the vector database")

def process_audio(audio_file):
    if audio_file is None:
        return "No audio file provided", None  # Handle case where no file is uploaded

    if isinstance(audio_file, str):
        audio_file_path = audio_file
    else:
        audio_file_path = audio_file.name

    wav_path = convert_to_wav(audio_file_path)
    if wav_path is None:
        return "Error converting audio file to WAV format", None

    text = audio_to_text(wav_path)
    if not text:
        return "No valid text extracted from audio", None

    try:
        audio_embedding = embed_text([text], model, tokenizer)[0]
        if audio_embedding.size == 0:
            return "Error generating embedding for the audio text", None
        
        distances, indices = index.search(np.array([audio_embedding]), k=2)
        relevant_texts = [texts[idx] for idx in indices[0]]
        combined_text = " ".join(relevant_texts)
        if len(combined_text) > 1000:
            combined_text = combined_text[:1000]

        if not combined_text.strip():
            return "No relevant information found in the PDFs", None
        prompt = (
      f"The user has described a child's behavior as follows: {combined_text}. "
     "Based on this description, provide 4 clear and actionable steps to manage the child's behavior directly. "
     "The response should focus only on practical actions for addressing the child's behavior " 
     "and make the response be no longer than 5 lines."
       )


        print(f"Prompt: {prompt}")  # Debugging line

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-70b-versatile",
        )
        response = chat_completion.choices[0].message.content
        output_file = "advice.mp3"
        output_path = text_to_speech(response, output_file)

        if output_path is None:
            return "Error generating speech output", None

        return response, output_path
    except Exception as e:
        print(f"Error in process_audio: {e}")
        return "An error occurred while processing the audio", None

# Define Gradio interface with custom colors

theme = gr.themes.Soft(
    neutral_hue=gr.themes.Color(c100="#fffdf5", c200="#fdfbee", c300="#f7e4cc", c400="#eac7b1", c50="#fffdf5", c500="#cc9e8e", c600="#bf9282", c700="#a47d70", c800="#9f7465", c900="#976c5e", c950="#7e5649"),
    secondary_hue="red",
    primary_hue=gr.themes.Color(c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd", c400="#5298b2", c50="#fffcf0", c500="#488ba6", c600="#487592", c700="#46748e", c800="#3e6785", c900="#22446a", c950="#002248"),
)

with gr.Blocks(
    theme=theme
) as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <img src="https://huggingface.co/spaces/Ayesha931/ChillMama3.0/blob/main/logo.jpeg" style="max-width: 50px; height: auto;" />
        </div>
        <h1 style='text-align: center; color: #3b718e; font-size: 48px; font-weight: bold;'>ChillMama</h1>
        <h3 style='text-align: center; color: #001f43; font-size: 24px;'>Helping Parents to Nurture Healthy Families</h3>
        """
    )
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            submit_btn = gr.Button("Submit")
    
    with gr.Row():
        with gr.Column():
            advice_output = gr.Textbox(label="Advice")
            advice_audio_output = gr.Audio(label="Advice Audio")
    
    # Bind the audio input and button to the process function
    submit_btn.click(fn=process_audio, inputs=audio_input, outputs=[advice_output, advice_audio_output])

# Launch the Gradio app with a public link
if __name__ == "__main__":
    demo.launch(share=True)  # Add share=True here to create a public link
