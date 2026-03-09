import os
import random
import shutil
from pathlib import Path

# --- ตั้งค่า Path ---
source_dir = "raw_data" # โฟลเดอร์ที่มีภาพและ .txt รวมกัน
dataset_dir = "solar_dataset"        # โฟลเดอร์ปลายทางที่จะสร้าง

# สร้างโครงสร้างโฟลเดอร์
for split in ['train', 'val']:
    for category in ['images', 'labels']:
        os.makedirs(os.path.join(dataset_dir, category, split), exist_ok=True)

# ดึงรายชื่อไฟล์ภาพทั้งหมด (ปรับนามสกุลได้ตามจริง)
image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)

# แบ่งสัดส่วน 80/20
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def copy_data(file_list, split_name):
    for img_file in file_list:
        base_name = os.path.splitext(img_file)[0]
        txt_file = base_name + ".txt"
        
        # ก๊อปปี้ภาพ
        shutil.copy(os.path.join(source_dir, img_file), 
                    os.path.join(dataset_dir, 'images', split_name, img_file))
        
        # ก๊อปปี้ Label (ถ้ามี)
        txt_path = os.path.join(source_dir, txt_file)
        if os.path.exists(txt_path):
            shutil.copy(txt_path, 
                        os.path.join(dataset_dir, 'labels', split_name, txt_file))

copy_data(train_files, 'train')
copy_data(val_files, 'val')
print(f"แบ่งข้อมูลสำเร็จ: Train {len(train_files)} ภาพ, Val {len(val_files)} ภาพ")