import os
import json
import logging

# إخفاء السجلات المزعجة
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

import chromadb
from chromadb.utils import embedding_functions

class KnowledgeRAG:
    """محرك الـ RAG المحلي لنظام مبصر باستعمال ChromaDB"""
    def __init__(self, json_path="mubser_knowledge.json", db_path="./chroma_db"):
        self.json_path = json_path
        self.db_path = db_path
        
        # نموذج Embeddings مجاني خفيف يدعم اللغة العربية
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # تهيئة قاعدة البيانات المحلية
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="mubser_hajj_knowledge",
            embedding_function=self.embedding_fn
        )
        
        # بناء الفهرس عند التشغيل لأول مرة
        self.load_data_into_vector_store()

    def load_data_into_vector_store(self):
        """تفكيك وقراءة ملف JSON المخصص لمُبصر وتحويله إلى Vector Store"""
        if not os.path.exists(self.json_path):
            print(f"⚠️ [RAG]: ملف المعرفة {self.json_path} غير موجود.")
            return

        if self.collection.count() > 0:
            return  # البيانات محمّلة مسبقاً

        print("📦 [RAG]: جاري بناء الفهرس الدلالي (Vector Store)...")
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ids = []
        documents = []
        metadatas = []

        sanctuary_info = data.get("makkah_sanctuary_info", {})
        all_items = []
        for category_key, items in sanctuary_info.items():
            if isinstance(items, list):
                for item in items:
                    item["category"] = category_key
                    all_items.append(item)

        for idx, item in enumerate(all_items):
            ids.append(item.get("id", f"doc_{idx}"))
            
            name_ar = item.get("name_ar", "")
            location_ar = item.get("location_ar", "")
            desc_ar = item.get("description_ar", "")
            fiqh_note = item.get("fiqh_note_ar", "")
            keywords = ", ".join(item.get("keywords", []))

            formatted_doc = (
                f"المعلم/الخدمة: {name_ar}\n"
                f"الموقع: {location_ar}\n"
                f"الوصف: {desc_ar}\n"
                f"ملاحظة فقهية: {fiqh_note}\n"
                f"كلمات مفتاحية: {keywords}"
            ).strip()

            documents.append(formatted_doc)
            metadatas.append({
                "category": item.get("category", "general"),
                "name_ar": name_ar
            })

        if documents:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✅ [RAG]: تم فهرسة {len(documents)} عنصر بنجاح.")

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        """الدالة المطلوبة لاسترجاع السياق بواسطة QAAgent"""
        if self.collection.count() == 0:
            return ""

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            documents = results.get("documents", [[]])[0]
            if not documents:
                return ""

            return "\n---\n".join(documents)
        except Exception as e:
            print(f"⚠️ [RAG Error]: {e}")
            return ""