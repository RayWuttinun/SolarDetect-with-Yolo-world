from ultralytics import YOLO
from tqdm import tqdm

# ฟังก์ชัน Callback ไว้ด้านบนสุดได้
def on_train_epoch_end(trainer):
    pbar.update(1)

# ✅ จุดสำคัญ: ต้องครอบโค้ดทำงานทั้งหมดไว้ใต้บรรทัดนี้
if __name__ == '__main__':
    
    # โหลดโมเดล
    model = YOLO('yolov8x-worldv2.pt')
    model.set_classes(["solar panel"])

    # ตั้งค่า Progress bar
    total_epochs = 100
    pbar = tqdm(total=total_epochs, desc="Training YOLO-World", bar_format="{l_bar}{bar}| Epoch {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # สั่ง Train
    results = model.train(
        data='solar.yaml',
        epochs=total_epochs,
        imgsz=512,
        batch=16,
        
        # 💡 แนะนำเพิ่มเติม: ถ้าใช้ Laptop ผมแนะนำให้ตั้ง workers ลดลงมาเหลือ 2 หรือ 4 นะครับ 
        # (ค่าเริ่มต้นคือ 8 อาจจะทำให้ CPU ทำงานหนักจนคอมค้างได้)
        workers=4, 

        freeze=10,
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        degrees=90.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
        project='solar_detection',
        name='yolo_world_tuning',
        device='0',
        verbose=False 
    )

    pbar.close()
    print("Training Complete!")