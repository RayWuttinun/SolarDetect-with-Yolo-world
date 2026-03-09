import gradio as gr
import numpy as np
import cv2
from PIL import Image
import os
import zipfile
import tempfile
import shutil
import atexit
from datetime import datetime

# ==========================================
# 🛑 ระบบจัดการไฟล์ชั่วคราว
# ==========================================
BASE_TEMP_DIR = tempfile.mkdtemp(prefix="solar_eva_temp_")

def cleanup_temp_dir():
    if os.path.exists(BASE_TEMP_DIR):
        shutil.rmtree(BASE_TEMP_DIR)
        print("🗑️ ล้างข้อมูลชั่วคราวทั้งหมดเรียบร้อยแล้ว")

atexit.register(cleanup_temp_dir)

# ==========================================
#  1. โหลด YOLO World Model
# ==========================================
try:
    from ultralytics import YOLOWorld
    MODEL_PATH = "best.pt"
    model = YOLOWorld(MODEL_PATH)
    print(f"✅ โหลด model สำเร็จ: {MODEL_PATH}")

    try:
        import clip as clip_module
        _orig_encode_text = clip_module.model.CLIP.encode_text
        def _patched_encode_text(self, text):
            device = self.token_embedding.weight.device
            return _orig_encode_text(self, text.to(device))
        clip_module.model.CLIP.encode_text = _patched_encode_text
    except Exception as e:
        print(f"⚠️ CLIP patch ไม่สำเร็จ: {e}")

except Exception as e:
    print(f"❌ โหลด model ไม่สำเร็จ: {e}")
    model = None

# ==========================================
#  2. Utilities & Evaluation Logic
# ==========================================
COLORS = [
    (251, 191, 36), (52, 211, 153), (96, 165, 250), (248, 113, 113),
    (167, 139, 250), (34, 211, 238), (251, 146, 60), (163, 230, 53)
]

def get_color(idx): return COLORS[idx % len(COLORS)]

def resolve_name(names, cls_id):
    if isinstance(names, dict): return names.get(cls_id, str(cls_id))
    elif isinstance(names, list): return names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
    return str(cls_id)

def draw_boxes(image_bgr, result):
    annotated = image_bgr.copy()
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            label = resolve_name(names, cls_id)
            color = get_color(cls_id)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 1, cv2.LINE_AA)
            
            detections.append([x1, y1, x2, y2])
    return annotated, detections

def bb_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def parse_yolo_label(filepath, img_w, img_h):
    gts = []
    if not os.path.exists(filepath): return gts
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x_center, y_center, w, h = map(float, parts[1:5])
                x1 = int((x_center - w/2) * img_w)
                y1 = int((y_center - h/2) * img_h)
                x2 = int((x_center + w/2) * img_w)
                y2 = int((y_center + h/2) * img_h)
                gts.append([x1, y1, x2, y2])
    return gts

# ==========================================
#  3. Core Functions
# ==========================================
def predict_single(image, prompts, conf_threshold, iou_threshold, max_det):
    if image is None: return None, "⚠️ กรุณาอัปโหลดรูปภาพ"
    if model is None: return image, "❌ โหลด model ไม่สำเร็จ"

    classes = [c.strip() for c in prompts.split(",") if c.strip()]
    if not classes: return image, "⚠️ กรุณากรอก Prompt"

    model.set_classes(classes)
    results = model.predict(Image.fromarray(image), conf=conf_threshold, iou=iou_threshold, max_det=max_det, verbose=False)
    
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_bgr, detections = draw_boxes(img_bgr, results[0])
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), f"### ✅ พบ {len(detections)} วัตถุ"

def run_eva_batch(image_files, label_files, prompts, conf_threshold, iou_threshold, max_det):
    if not image_files: return [], "⚠️ กรุณาอัปโหลดโฟลเดอร์รูปภาพ", gr.update(visible=False)
    if model is None: return [], "❌ โหลด model ไม่สำเร็จ", gr.update(visible=False)

    classes = [c.strip() for c in prompts.split(",") if c.strip()]
    if not classes: return [], "⚠️ กรุณากรอก Prompt", gr.update(visible=False)

    model.set_classes(classes)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_TEMP_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # กรองเฉพาะไฟล์ภาพ
    img_paths = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # จับคู่ Label
    label_map = {}
    has_labels = False
    if label_files:
        has_labels = True
        for lf in label_files:
            if lf.endswith('.txt'):
                basename = os.path.splitext(os.path.basename(lf))[0]
                label_map[basename] = lf

    gallery_images = []
    
    # ตัวแปรสถิติ
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    total_mae = 0
    total_imgs = len(img_paths)

    for filepath in img_paths:
        img_bgr = cv2.imread(filepath)
        if img_bgr is None: continue
        h, w = img_bgr.shape[:2]

        results = model.predict(img_bgr, conf=conf_threshold, iou=iou_threshold, max_det=max_det, verbose=False)
        annotated_bgr, preds = draw_boxes(img_bgr, results[0])
        
        filename = os.path.basename(filepath)
        basename = os.path.splitext(filename)[0]
        out_path = os.path.join(run_dir, f"pred_{filename}")
        cv2.imwrite(out_path, annotated_bgr)
        gallery_images.append((cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), f"{filename} ({len(preds)} objs)"))

        # --- คำนวณ Confusion Matrix ถ้ายูสเซอร์ใส่โฟลเดอร์ Label มา ---
        if has_labels:
            gts = parse_yolo_label(label_map.get(basename, ""), w, h)
            total_mae += abs(len(preds) - len(gts))

            if len(gts) == 0 and len(preds) == 0:
                total_TN += 1
            else:
                # คำนวณ TP, FP, FN ด้วย IoU
                matched_gt = set()
                img_TP = 0
                for p_box in preds:
                    best_iou, best_gt_idx = 0, -1
                    for idx, gt_box in enumerate(gts):
                        if idx in matched_gt: continue
                        iou = bb_iou(p_box, gt_box)
                        if iou > iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                    
                    if best_gt_idx >= 0:
                        matched_gt.add(best_gt_idx)
                        img_TP += 1
                
                img_FP = len(preds) - img_TP
                img_FN = len(gts) - img_TP
                
                total_TP += img_TP
                total_FP += img_FP
                total_FN += img_FN

    # --- สร้าง Dashboard Text ---
    if has_labels:
        total_eval = total_TP + total_FP + total_FN + total_TN
        accuracy = (total_TP + total_TN) / total_eval if total_eval > 0 else 0
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0
        mae_avg = total_mae / total_imgs if total_imgs > 0 else 0

        dashboard_text = f"""```text
=======================================================
           CONFUSION MATRIX (ระดับภาพ/object)
=======================================================
                         Predicted +   Predicted -
  Actual +  (has object)      TP = {total_TP:<8} FN = {total_FN}
  Actual -  (no object)       FP = {total_FP:<8} TN = {total_TN}
  * TN นับระดับภาพ: ภาพที่ไม่มี GT และ model ไม่ detect เลย
=======================================================
  Accuracy    : {accuracy:.4f}  ({accuracy*100:.2f}%)
  Precision   : {precision:.4f}  ({precision*100:.2f}%)
  Recall      : {recall:.4f}  ({recall*100:.2f}%)
  F1 Score    : {f1:.4f}  ({f1*100:.2f}%)
  Specificity : {specificity:.4f}  ({specificity*100:.2f}%)
=======================================================
            MAE Count
=======================================================
  MAE Count   : {mae_avg:.4f}  (avg |pred_n - gt_n| across {total_imgs} images)
```"""
    else:
        dashboard_text = f"```text\n[ระบบประมวลผลไปทั้งหมด {total_imgs} ภาพ]\n*หากต้องการดู Confusion Matrix กรุณาอัปโหลดโฟลเดอร์ Label ด้วย*\n```"

    # --- สร้าง ZIP ---
    zip_path = os.path.join(BASE_TEMP_DIR, f"EVA_Result_{run_id}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(run_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.basename(file_path))
        report_path = os.path.join(run_dir, "eva_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(dashboard_text.replace("```text\n", "").replace("\n```", ""))
        zipf.write(report_path, "eva_report.txt")

    return gallery_images, dashboard_text, gr.update(value=zip_path, visible=True)

# ==========================================
#  4. CSS Stylesheet 
# ==========================================
CSS = """
body, .gradio-container { background: #0d0f14 !important; font-family: 'Inter', sans-serif !important; color: #e8eaf0 !important; }
#app-header { text-align: center; padding: 20px 0; border-bottom: 1px solid #252d3d; margin-bottom: 20px; }
#app-header h1 { font-size: 2rem; color: #f5a623; margin: 0; }
.section-label { font-size: 0.75rem; font-weight: bold; letter-spacing: 2px; text-transform: uppercase; color: #f5a623; border-bottom: 1px solid #252d3d; margin-bottom: 10px; }
textarea, input[type="text"] { background: #0a0c10 !important; border: 1px solid #252d3d !important; border-radius: 8px !important; color: white !important; }
.gallery-item img { border-radius: 8px; }
footer { display: none !important; }
"""

# ==========================================
#  5. Gradio UI
# ==========================================
with gr.Blocks(theme=gr.themes.Base(), css=CSS) as demo:
    gr.HTML("""
    <div id="app-header">
        <h1>☀️ SOLAR DETECT</h1>
        <p>YOLO WORLD · FINE-TUNED · SOLAR PANEL DETECTION</p>
    </div>
    """)

    with gr.Tabs():
        
        # ------------------------------------
        # TAB 1: Single Detect
        # ------------------------------------
        with gr.Tab("⚡ Single Image"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.HTML('<div class="section-label">Input</div>')
                    prompt_1 = gr.Textbox(label="Prompt", placeholder="solar panel, crack...", lines=2)
                    img_in = gr.Image(label="Image", type="numpy", height=300)
                    with gr.Accordion("⚙️ Settings", open=False):
                        conf_1 = gr.Slider(0.01, 1.0, 0.25, label="Conf")
                        iou_1 = gr.Slider(0.1, 1.0, 0.45, label="IoU")
                        max_1 = gr.Slider(1, 300, 100, label="Max Det")
                    btn_single = gr.Button("⚡ DETECT", variant="primary")
                    
                with gr.Column(scale=5):
                    gr.HTML('<div class="section-label">Output</div>')
                    img_out = gr.Image(label="Result", height=300, interactive=False)
                    txt_out = gr.Markdown("ผลลัพธ์จะแสดงที่นี่")

            btn_single.click(predict_single, [img_in, prompt_1, conf_1, iou_1, max_1], [img_out, txt_out])

        # ------------------------------------
        # TAB 2: EVA Mode (Batch Directory)
        # ------------------------------------
        with gr.Tab("📊 EVA Mode (Batch)"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.HTML('<div class="section-label">1. Upload Directories</div>')
                    
                    # อนุญาตให้อัปโหลดทั้งโฟลเดอร์ได้
                    img_dir = gr.File(label="📂 1. อัปโหลดโฟลเดอร์ Image", file_count="directory", type="filepath")
                    lbl_dir = gr.File(label="📂 2. อัปโหลดโฟลเดอร์ Label (.txt) [ไม่บังคับ]", file_count="directory", type="filepath")
                    
                    prompt_2 = gr.Textbox(label="Prompt", placeholder="solar panel, crack...", lines=2)
                    with gr.Accordion("⚙️ Settings", open=False):
                        conf_2 = gr.Slider(0.01, 1.0, 0.25, label="Conf")
                        iou_2 = gr.Slider(0.1, 1.0, 0.45, label="IoU Match Threshold")
                        max_2 = gr.Slider(1, 300, 100, label="Max Det")
                    
                    btn_eva = gr.Button("▶️ RUN EVA BATCH", variant="primary", size="lg")
                    download_zip = gr.File(label="📥 ดาวน์โหลดไฟล์ ZIP", visible=False)

                with gr.Column(scale=6):
                    gr.HTML('<div class="section-label">2. EVA Dashboard & Gallery</div>')
                    eva_dashboard = gr.Markdown("📊 **Dashboard:** รอการอัปโหลดโฟลเดอร์และรันโมเดล...")
                    eva_gallery = gr.Gallery(label="🖼️ ดูผลลัพธ์รายรูป", columns=3, height=450)

            btn_eva.click(
                fn=run_eva_batch,
                inputs=[img_dir, lbl_dir, prompt_2, conf_2, iou_2, max_2],
                outputs=[eva_gallery, eva_dashboard, download_zip]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)