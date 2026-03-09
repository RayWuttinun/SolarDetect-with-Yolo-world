FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ป้องกัน python cache
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ติดตั้ง system packages ที่ opencv ต้องใช้
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# copy dependency
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]