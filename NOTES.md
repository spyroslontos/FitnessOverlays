# FitnessOverlays – Dev Cheatsheet

## Local Development

```bash
python -m venv .fitnessoverlays-venv
source .fitnessoverlays-venv/bin/activate
cp -n .env.example .env
pip install -r requirements.txt
python server.py
```

## Docker (Production Parity)

```bash
docker build -t fitnessoverlays-app .

docker run -p 8000:8000 \
  --name fitnessoverlays-web \
  --env-file .env \
  -d fitnessoverlays-app

# Hot reload against local files
docker run -p 8000:8000 \
  --name fitnessoverlays-web-dev \
  -v "/Users/spyros/projects/FitnessOverlays:/app" \
  --env-file .env \
  -d fitnessoverlays-app

docker logs -f fitnessoverlays-web
docker stop fitnessoverlays-web && docker rm fitnessoverlays-web
```

## Tailwind

```bash
npx @tailwindcss/cli -i ./static/css/input.css -o ./static/css/tailwind.css --watch
npx @tailwindcss/cli -i ./static/css/input.css -o ./static/css/tailwind.css --minify
```

## Dependencies

```bash
pip install -r requirements.txt
pip freeze > requirements-frozen.txt
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
