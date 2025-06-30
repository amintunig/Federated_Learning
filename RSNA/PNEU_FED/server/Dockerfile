FROM python:3.9-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
WORKDIR /app

COPY . /app/
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt
COPY server/ server/
COPY model/ cnn/
COPY config/ setting/
COPY client/ dataset/
CMD ["python", "server/server.py"]
