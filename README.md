# ChillMama: Multi-Modal GenAI Parenting & Behavior Assistant 🚀

An intelligent, voice-driven multi-modal application engineered to assist mothers during high-stress parenting moments. By providing immediate, context-aware guidance, ChillMama serves as both an active intervention tool and an awareness platform to promote positive parenting, reduce household conflict, and ensure children are raised in a supportive environment free from childhood trauma.

🔗 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Ayesha931/ChillMama3.0)** | 📂 **[Data Ingestion Directory (Google Drive Download)](https://drive.google.com/file/d/1UpzE2_sv7UFbuev_C9Nj7nDC25eF7qsO/view?usp=sharing)**

---

## 🧠 Architectural Workflow & Core Features
- **Voice-to-Text Pipeline:** Integrates automated audio processing utilizing `pydub` format normalization paired with the `speech_recognition` engine to parse and transcribe microphone voice inputs seamlessly.
- **Localized Semantic Ingestion (RAG):** Extracts underlying reference text from comprehensive pediatric and positive parenting guidelines via `PyMuPDF` (`fitz`), transforming text matrices into 768-dimensional embeddings through a local `bert-base-uncased` transformer model.
- **Dense Vector Search Engine:** Indexes historical document tensors using a localized `FAISS` instance, running real-time L2 distance similarity queries to fetch expert parenting context for the LLM prompt payload.
- **Ultra-Fast Synthesis:** Routes context-augmented prompts to the high-performance **Groq Cloud API** running open-source model infrastructure (`llama-3.3-70b-versatile`) to generate rapid, structured advice.
- **Text-to-Voice Output (TTS):** Automatically synthesizes the AI's structural text recommendations back into clean, human-playable `.mp3` audio files via Google Text-to-Speech (`gTTS`).

---

## 🖥️ User Interface & Accessibility Stack
The frontend features a clean, accessibility-focused interface engineered using the **Gradio** ecosystem. It supports seamless cross-platform microphone inputs, real-time response generation, and dual text/audio playback components designed for low-friction operation during stressful scenarios.

---

## 📁 Repository Inventory
- `app.py`: Core application architecture containing audio translation pipelines, transformer mean pooling calculations, API configurations, and layout states.
- `pdfsforRAG/`: Target context folder where reference files and expert parenting books are parsed (Note: Due to file size limits, the full production document dataset is hosted externally on Google Drive).
