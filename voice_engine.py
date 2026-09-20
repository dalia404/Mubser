import os
import time
import json
import queue
import asyncio
import threading
import edge_tts
import pygame
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ----------------------------------------------------
# 1. إعداد محرك تشغيل الصوت (TTS - Text to Speech)
# ----------------------------------------------------
try:
    if not pygame.mixer.get_init():
        pygame.mixer.init()
except Exception:
    pass

VOICE = "ar-EG-SalmaNeural"

async def _generate_speech(text, output_file):
    short_text = text[:300] if len(text) > 300 else text
    communicate = edge_tts.Communicate(short_text, VOICE)
    await communicate.save(output_file)

def _play_audio_task(text):
    if not text or not text.strip():
        return

    temp_file = f"temp_{int(time.time() * 1000)}.mp3"
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.unload()

        # إنشاء Event Loop مستقل لكل عملية نطق لمنع تعارض Asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate_speech(text, temp_file))
        loop.close()

        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

        pygame.mixer.music.unload()
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception as e:
        print(f"⚠️ تنبيه تشغيل الصوت: {e}")

def speak_offline(text):
    """دالة نطق الصوت في خيط مستقل (Thread) لمنع تجمد الواجهة"""
    threading.Thread(target=_play_audio_task, args=(text,), daemon=True).start()


# ----------------------------------------------------
# 2. إعداد محرك الاستماع والتحويل من صوت لنص (STT)
# ----------------------------------------------------
class VoiceListener:
    def __init__(self, model_path=None, device_index=None):
        self.q = queue.Queue()
        self.model_loaded = False
        self.device_index = device_index
        
        # تحديد المسار المطلق لمجلد model في نفس دليل المشروع
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "model")
        
        # التأكد من وجود مجلد الموديل المحلي لـ Vosk
        if os.path.exists(model_path):
            try:
                self.model = Model(model_path)
                self.recognizer = KaldiRecognizer(self.model, 16000)
                self.model_loaded = True
                print(f"✅ [Vosk Engine]: تم تحميل نموذج التعرف الصوتي بنجاح من المسار:\n   └─ {model_path}")
            except Exception as e:
                print(f"⚠️ [Vosk Engine] خطأ أثناء تحميل نموذج Vosk: {e}")
        else:
            print(f"⚠️ [Vosk Engine]: لم يتم العثور على مجلد 'model' في المسار التالي:\n   └─ {model_path}")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            pass
        self.q.put(bytes(indata))

    def listen_once(self, timeout=4):
        """استماع محلي سريع جداً من الميكروفون"""
        if not self.model_loaded:
            print("⚠️ [Vosk Engine]: تعذر الاستماع لأن النموذج غير محمل.")
            return None

        # إفراغ البافر لتجنب القراءات الصوتية القديمة
        with self.q.mutex:
            self.q.queue.clear()

        try:
            with sd.RawInputStream(samplerate=16000, blocksize=8000, device=self.device_index,
                                   dtype='int16', channels=1, callback=self._audio_callback):
                
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        data = self.q.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            print(f"🎙️ [Vosk Offline]: {text}")
                            return text
                
                final_res = json.loads(self.recognizer.FinalResult())
                text = final_res.get("text", "").strip()
                return text if text else None

        except Exception as e:
            print(f"⚠️ خطأ استماع محلي: {e}")
            return None