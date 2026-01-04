# FitnessOverlays – Dev Cheatsheet

## Local Development

```bash
python -m venv .fitnessoverlays-venv
source .fitnessoverlays-venv/bin/activate
cp -n .env.example .env
pip install -r requirements.txt
python server.py
```

## Dependencies

```bash
pip install -r requirements.txt
pip freeze > requirements-frozen.txt
```

## Docker (Production Parity)

```bash
docker build -t fitnessoverlays .

docker run -p 8000:8000 \
  --env-file .env \
  --name fitnessoverlays \
  -d fitnessoverlays

# Docker hot reload
docker run -p 8000:8000
  --name fitnessoverlays-dev
  -v "/Users/spyros/projects/FitnessOverlays:/app"
  -w /app
  --env-file .env
  -d python:3.11-slim
  bash -c "pip install -r requirements.txt &&
  npx tailwindcss -i static/css/input.css -o static/css/tailwind.css --watch &
  FLASK_APP=server.py FLASK_ENV=development flask run --host=0.0.0.0 --port=8000"

docker logs -f fitnessoverlays-dev
docker stop fitnessoverlays-dev && docker rm fitnessoverlays-dev
```

## Tailwind

```bash
npx @tailwindcss/cli -i ./static/css/input.css -o ./static/css/tailwind.css --watch

npx @tailwindcss/cli -i ./static/css/input.css -o ./static/css/tailwind.css --minify
```

## Strava Webhooks

```bash
curl -X GET  "https://www.strava.com/api/v3/push_subscriptions?client_id=ID&client_secret=SECRET"

curl -X DELETE "https://www.strava.com/api/v3/push_subscriptions/SUB_ID?client_id=ID&client_secret=SECRET"

curl -X POST https://www.strava.com/api/v3/push_subscriptions \
  -F client_id=ID \
  -F client_secret=SECRET \
  -F callback_url=https://fitnessoverlays.com/webhook \
  -F verify_token=TOKEN
```
