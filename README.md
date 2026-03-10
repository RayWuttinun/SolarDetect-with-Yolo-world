# SolarDetect-with-Yolo-world ☀️🛰️

โปรเจกต์นี้เป็นการประยุกต์ใช้ **YOLO-World** (Vision Language Model) เพื่อตรวจจับ "แผงโซลาร์เซลล์" (Solar Panels) จากภาพถ่ายดาวเทียม โดยใช้ Zero-shot ได้ทันทีโดยไม่ต้องเทรนโมเดลใหม่ตั้งแต่ต้น

หลังจาก Git Clone เเล้วสามารถรับโมเดลที่เทรนเเล้วได้มาวางในโฟลเดอร์ SolarDetect-with-Yolo-world ได้ที่ลิ้งนี้
link
จากนั้นสามารถ
```bash
docker build -t solar-detection-app .
docker run -d -p 7860:7860 --name solar-container solar-detection-app
http://localhost:7860/
```
วิธีตามปกติ
**1. Clone Repository**
```bash
git clone [https://github.com/RayWuttinun/SolarDetect-with-Yolo-world.git](https://github.com/RayWuttinun/SolarDetect-with-Yolo-world.git)
cd SolarDetect-with-Yolo-world
```
**2. Set Up Docker**
Build Docker Container
```bash
docker build -t solar-detection-app .
```
**3. Train.py**
รันให้ได้โมเดล runs\detect\solar_detection\yolo_world_tuning\weights\best.pt เเล้ว copy มาวางนอกโฟลเดอร์ runs
```bash
uv run python Train.py
```
**4. Enter Web ***
```bash
docker run -d -p 7860:7860 --name solar-container solar-detection-app
http://localhost:7860/
```
