FROM python:3.9-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
WORKDIR /app

COPY requirements.txt .
#COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python", "client.py", "--cid =1"]