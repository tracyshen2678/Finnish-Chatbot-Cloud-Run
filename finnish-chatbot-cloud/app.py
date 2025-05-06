import re
import os
import io
import time
import numpy as np
import torch
import ffmpeg
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import uvicorn
import nest_asyncio
from pathlib import Path

nest_asyncio.apply()

os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

speech_recognizer = pipeline(
    "automatic-speech-recognition",
    model="Finnish-NLP/whisper-large-finnish-v3",
    device=device
)

gpt2_tokenizer = AutoTokenizer.from_pretrained("TracyShen301/myFinnishChatbot")
gpt2_model = AutoModelForCausalLM.from_pretrained("TracyShen301/myFinnishChatbot").to(device)

AUDIO_DIR = "/tmp/audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

hardcoded_qa = {
    "kuka sinä olet": "Olen sinun suomen kielen oppimisrobotti.",
    "olen tänään todella väsynyt": "Lepää hyvin, huomenna on parempi päivä.",
    "missä on lähin ruokakauppa": "Käänny oikealle, kauppa on 200 metrin päässä.",
    "miten pääsen kirjastoon": "Suoraan eteenpäin ja toinen katu vasemmalle.",
    "millainen sää on tänään": "Aurinkoista ja lämmintä, noin 22 astetta.",
    "missä voin pelata jalkapalloa": "Keskuspuistossa on hyvät kentät.",
    "mitä ruokaa suosittelet": "Kokeile lohikeittoa, se on paikallinen erikoisuus.",
    "mitä kuuluu": "Hyvää, kiitos kysymästä.",
    "miksi suomi on niin vaikeaa": "Harjoittelu tekee mestarin, älä luovuta.",
    "paljonko tämä maksaa": "Se maksaa 15 euroa.",
    "milloin bussi tulee": "Seuraava bussi tulee 10 minuutin päästä.",
    "voinko varata ajan huomiselle": "Kyllä, meillä on aikoja vapaana klo 14 ja 16.",
    "onko teillä kofeiiniton kahvi": "Kyllä, haluatko sen maidolla?",
    "tarvitsen särkylääkettä missä apteekki on": "Apteekki on kauppakeskuksen toisessa kerroksessa.",
    "monelta on aamiainen": "Aamiainen tarjoillaan 7-10 välillä.",
    "voiko täällä uida": "Kyllä, vesi on puhdasta ja lämmintä."
}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Finnish Chatbot API!"}

@app.post("/process_audio/")
async def process_audio(audio: UploadFile = File(...)):
    try:
        input_path = os.path.join(AUDIO_DIR, "input.mp3")
        with open(input_path, "wb") as f:
            f.write(await audio.read())

        out, _ = (
            ffmpeg.input(input_path)
            .output('pipe:1', format='wav', ac=1, ar=16000)  
            .run(capture_stdout=True, capture_stderr=True)
        )

        audio_data, samplerate = sf.read(io.BytesIO(out))
        audio_data = (audio_data * 32767).astype(np.int16) 

        result = speech_recognizer({"raw": audio_data, "sampling_rate": 16000})
        input_text = result["text"].strip()

        input_text = input_text.strip("\"""")

        input_text_clean = re.sub(r"[^\w\s]", "", input_text).lower()

        if input_text_clean in hardcoded_qa:
            response_text = hardcoded_qa[input_text_clean]
        else:
            formatted_input_text = f"User: {input_text}"
            inputs = gpt2_tokenizer.encode(formatted_input_text, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(inputs)
            outputs = gpt2_model.generate(
                inputs,
                max_length=150,
                pad_token_id=50256,
                attention_mask=attention_mask
            )
            response_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Bot:" in response_text:
                response_text = response_text.split("Bot:")[-1].strip()

            match = re.search(r"([^.?!]*[.?!])", response_text)
            if match:
                response_text = match.group(1).strip()

        tts = gTTS(text=response_text, lang="fi")
        unique_filename = f"response_{int(time.time() * 1000)}.mp3"
        output_path = os.path.join(AUDIO_DIR, unique_filename)
        tts.save(output_path)  
        
        return {
            "transcription": input_text,
            "response": response_text,
            "audio_url": f"/download_response/{unique_filename}"
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/download_response/{filename}")
async def download_response(filename: str):
    return FileResponse(
        os.path.join(AUDIO_DIR, filename),
        media_type="audio/mpeg",
        filename=filename
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

