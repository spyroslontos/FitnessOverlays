FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install

COPY requirements-frozen.txt .
RUN pip install --no-cache-dir -r requirements-frozen.txt

COPY . .
RUN npx @tailwindcss/cli -i ./static/css/input.css -o ./static/css/tailwind.css --minify && \
    rm -rf node_modules package-lock.json

EXPOSE 8000
CMD ["gunicorn", "-w", "1", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "server:app"]
