# SolarDetect-with-Yolo-world ☀️🛰️

โปรเจกต์นี้เป็นการประยุกต์ใช้ **YOLO-World** (Vision Language Model) เพื่อตรวจจับ "แผงโซลาร์เซลล์" (Solar Panels) จากภาพถ่ายดาวเทียม โดยใช้ Zero-shot ได้ทันทีโดยไม่ต้องเทรนโมเดลใหม่ตั้งแต่ต้น



**1. Clone Repository**
```bash
git clone [https://github.com/RayWuttinun/SolarDetect-with-Yolo-world.git](https://github.com/RayWuttinun/SolarDetect-with-Yolo-world.git)
cd SolarDetect-with-Yolo-world
```
**2. Set Up Docker**
Build Docker Container
```bash
docker build -t solar-detection-app .
docker run -d -p 7860:7860 --name solar-container solar-detection-app
```
**3. Enter Web ***
```bash
http://localhost:7860/
```
