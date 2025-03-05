import re
import os
import io
import numpy as np
import torch
import ffmpeg
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS

# 设置缓存目录为 /tmp
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

# 初始化应用
app = FastAPI()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 语音识别 Pipeline
speech_recognizer = pipeline(
    "automatic-speech-recognition",
    model="Finnish-NLP/whisper-large-finnish-v3",
    device=device
)

# 对话模型
gpt2_tokenizer = AutoTokenizer.from_pretrained("TracyShen301/myFinnishChatbot")
gpt2_model = AutoModelForCausalLM.from_pretrained("TracyShen301/myFinnishChatbot").to(device)

# 文件存储配置，使用 /tmp/audio_files 目录
AUDIO_DIR = "/tmp/audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Finnish Chatbot API!"}


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # 保存原始音频
        input_path = os.path.join(AUDIO_DIR, "input.mp3")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # 音频预处理
        out, _ = (
            ffmpeg.input(input_path)
            .output('pipe:1', format='wav', ac=1, ar=16000)  # 强制16kHz单声道
            .run(capture_stdout=True, capture_stderr=True)
        )

        # 读取并转换音频格式
        audio_data, samplerate = sf.read(io.BytesIO(out))
        audio_data = (audio_data * 32767).astype(np.int16)  # 转换为16位整型

        # 语音识别 - 使用正确的字典键名格式
        result = speech_recognizer(
            {"raw": audio_data, "sampling_rate": 16000}
        )
        input_text = result["text"]
        formatted_input_text = f"User: {input_text}"

        # 生成对话响应
        inputs = gpt2_tokenizer.encode(formatted_input_text, return_tensors="pt").to(device)
        # 创建attention mask
        attention_mask = torch.ones_like(inputs)
        # 生成输出
        outputs = gpt2_model.generate(
            inputs,
            max_length=150,
            pad_token_id=50256,
            attention_mask=attention_mask
        )
        response_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 处理 GPT-2 生成的文本
        if "Bot:" in response_text:
            response_text = response_text.split("Bot:")[-1].strip()  # 只保留 "Bot:" 后的部分

        # 提取第一句话
        match = re.search(r"([^.?!]*[.?!])", response_text)
        if match:
            response_text = match.group(1).strip()

        # 语音合成
        tts = gTTS(text=response_text, lang="fi")
        output_path = os.path.join(AUDIO_DIR, "response.mp3")
        tts.save(output_path)

        return {
            "transcription": input_text,
            "response": response_text,
            "audio_url": "/download_response/"
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/download_response/")
async def download_response():
    return FileResponse(
        os.path.join(AUDIO_DIR, "response.mp3"),
        media_type="audio/mpeg",
        filename="response.mp3"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
