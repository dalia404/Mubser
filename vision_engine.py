import cv2
import numpy as np
from ultralytics import YOLO

try:
    from agents import VisionAgent as BaseVisionAgent
except ImportError:
    class BaseVisionAgent:
        pass


class VisionProcessor(BaseVisionAgent):
    """محرك الرؤية المحوسبة المدمج واستخراج التحديدات لـ Sensor Fusion"""
    def __init__(self, model_path="yolov8n.pt"):
        super().__init__()
        try:
            self.model = YOLO(model_path)
            print("👁️ تم تحميل نموذج YOLOv8 ومحرك الرؤية بنجاح")
        except Exception as e:
            print(f"⚠️ تحذير: تعذر تحميل نموذج YOLOv8 ({e})، سيتم استخدام وضع الاحتياطي.")
            self.model = None

    def extract_detections(self, frame):
        """استخراج العوائق والمسافات الأولية وتمريرها لمحرك دمج الحساسات (Sensor Fusion)"""
        if frame is None or self.model is None:
            return []

        h, w, _ = frame.shape
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])

            # تصفية العوائق المهمة للسلامة والملاحة
            if label in ['person', 'chair', 'table', 'sofa', 'door', 'wheelchair'] and conf > 0.4:
                bbox = box.xyxy[0].cpu().numpy()
                box_h = bbox[3] - bbox[1]
                
                # تقدير مبدئي للمسافة بناءً على نسبة ارتفاع صندوق التحديد بالنسبة للإطار
                ratio = box_h / h
                est_dist = round((1.0 / (ratio + 1e-5)) * 0.35, 2)
                est_dist = min(est_dist, 5.0)

                detections.append((label, bbox, est_dist, conf))

        return detections

    def process_frame(self, frame):
        """معالجة الفريم ورسم صندوق التحديدات والتنبيهات للواجهة المباشرة"""
        if frame is None:
            return None

        if self.model is not None:
            results = self.model(frame, verbose=False)[0]
            annotated_frame = results.plot()
        else:
            annotated_frame = frame.copy()

        # إضافة الترويسة التوضيحية لنظام مبصر
        cv2.putText(
            annotated_frame, 
            "Mubsir Vision AI - Active", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        return annotated_frame

    # دوال احتياطية وتوافقية للخدمات الصوتية والمعالم والذكاء الجمعي
    def detect_landmark(self):
        return "المسجد الحرام - الكعبة المشرفة"

    def sync_swarm_data(self):
        return "منطقة صحن المطاف: كثافة متوسطة، الحركة انسيابية"


# ربط الأسماء المستعارات لتجنب أي Break في الاستيراد في باقي الملفات
VisionAgent = VisionProcessor
VisionEngine = VisionProcessor