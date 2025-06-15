from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import torch
import gc
import fitz  # PyMuPDF
from KOKORO.models import build_model
from KOKORO.utils import tts



app = Flask(__name__)
app.secret_key = "kokoro_secret"

UPLOAD_FOLDER = "KOKORO/voices"
OUTPUT_FOLDER = "static/audio"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")
MODEL = build_model('./KOKORO/kokoro-v0_19.pth', device)
current_model = "kokoro-v0_19.pth"

# ---------- UTILITY FUNCTIONS ----------

def update_model(model_name):
    global MODEL, current_model
    if current_model == model_name:
        return
    model_path = f"./KOKORO/{model_name}"
    if model_name == "kokoro-v0_19-half.pth":
        model_path = f"./KOKORO/fp16/{model_name}"
    del MODEL
    gc.collect()
    torch.cuda.empty_cache()
    MODEL = build_model(model_path, device)
    current_model = model_name

def chunk_text(text, chunk_size=400):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def tts_maker(text, voice_name, speed, save_path):
    global MODEL
    return tts(
        MODEL, device, text, voice_name, speed=speed,
        trim=0.0, pad_between_segments=0,
        output_file=save_path,
        remove_silence=False, minimum_silence=50
    )

# ---------- ROUTES ----------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/text.html", methods=["GET", "POST"])
def text_to_audio():
    voice_list = sorted([
        os.path.splitext(f)[0] for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pt')
    ], key=len)

    model_list = ["kokoro-v0_19.pth", "kokoro-v0_19-half.pth"]
    audio_files = []

    if request.method == "POST":
        text = request.form["text"]
        voice = request.form["voice"]
        model = request.form["model"]
        speed = float(request.form["speed"])

        update_model(model)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            out_path = os.path.join(OUTPUT_FOLDER, f"episode_{i+1}.wav")
            tts_maker(chunk, voice, speed, out_path)
            audio_files.append(os.path.basename(out_path))

    return render_template("text.html", voices=voice_list, models=model_list, audio_files=audio_files)

@app.route("/audio/<filename>")
def get_audio(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/generate_chunk", methods=["POST"])
def generate_chunk():
    data = request.get_json()
    text = data["text"]
    voice = data["voice"]
    model = data["model"]
    speed = float(data["speed"])
    index = data["index"]

    update_model(model)

    filename = f"episode_{index + 1}.wav"
    out_path = os.path.join(OUTPUT_FOLDER, filename)
    tts_maker(text, voice, speed, out_path)

    return {"filename": filename, "index": index}

@app.route("/pdf.html", methods=["GET"])
def show_pdf_page():
    voice_list = sorted([
        os.path.splitext(f)[0] for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pt')
    ], key=len)

    model_list = ["kokoro-v0_19.pth", "kokoro-v0_19-half.pth"]
    return render_template("pdf.html", voices=voice_list, models=model_list)

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    file = request.files.get("pdf")
    if not file or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Invalid PDF"}), 400

    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 500

    chunks = chunk_text(full_text)
    return jsonify({"chunks": chunks})






if __name__ == "__main__":
    app.run(debug=True, port=8080)



