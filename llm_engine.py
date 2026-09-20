import json
import os
import requests
from config import OLLAMA_URL, OLLAMA_MODEL_NAME, SYSTEM_PROMPT, JSON_PATH


def is_ollama_available(timeout=1.5):
    """فحص سريع (أقل من ثانيتين) هل Ollama شغال محلياً قبل ما نحاول نرسل سؤال كامل.
    مهم عشان ما نستنى 60 ثانية (مهلة find_answer) لو Ollama أصلاً مش شغال."""
    try:
        base_url = OLLAMA_URL.replace("/api/generate", "")
        response = requests.get(base_url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False

def search_relevant_context(user_query):
    """البحث في ملف JSON عن الجزء المتعلق بسؤال المستخدم فقط لتسريع الاستجابة"""
    if not os.path.exists(JSON_PATH):
        return ""
    
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        makkah_info = data.get("makkah_sanctuary_info", {})
        query_lower = user_query.lower().strip()
        matched_info = []

        # البحث داخل جميع الأقسام (kaaba_landmarks, saai_landmarks, services_and_accessibility)
        for category, items in makkah_info.items():
            if isinstance(items, list):
                for item in items:
                    keywords = [k.lower() for k in item.get("keywords", [])]
                    # المطابقة مع الكلمات المفتاحية أو المعرف
                    if any(kw in query_lower for kw in keywords) or item.get("id") in query_lower:
                        matched_info.append(item)

        if matched_info:
            return json.dumps(matched_info, ensure_ascii=False)
            
    except Exception as e:
        print(f"❌ خطأ في قراءة ملف JSON: {e}")
        
    return ""

def find_answer(user_query):
    if not user_query or not str(user_query).strip():
        return "لم أسمعك بوضوح، تفضل بإعادة سؤالك."

    # جلب السياق من الـ JSON إن وجد
    context = search_relevant_context(user_query)

    # بناء الـ Prompt باللغة العربية بالكامل
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"تعليمات حازمة: الإجابة يجب أن تكون باللغة العربية الفصحى حصراً وبأسلوب واضح ومختصر يناسب النطق الصوتي. يمنع استخدام اللغة الإنجليزية مطلقاً.\n\n"
        f"السياق المتاح:\n{context if context else 'لا يوجد سياق محلي، أجب بناءً على معرفتك الدينية والمكانية بالحرم.'}\n\n"
        f"سؤال المستخدم: {user_query}\n"
        f"إجابة مُبصر (بالعربية):"
    )

    try:
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 250
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return "عذراً، حدث خطأ في الاتصال بنظام الذكاء الاصطناعي."

    except requests.exceptions.Timeout:
        print("❌ انتهت مهلة الاستجابة من Ollama")
        return "عذراً، استغرقت الإجابة وقتاً طويلاً. يرجى المحاولة مرة أخرى."
    except Exception as e:
        print(f"❌ Ollama Error: {e}")
        return "سيرفر الذكاء الاصطناعي غير متصل حالياً."