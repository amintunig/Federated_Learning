FROM python:3.9-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# ✅ IMPORTANT
COPY client/ client/
COPY model/ model/

CMD ["python", "client/client.py"]
