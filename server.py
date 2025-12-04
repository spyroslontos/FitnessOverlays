from flask import Flask, render_template, request, redirect, session, url_for, jsonify, Response, make_response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
import os
import httpx
import logging
from logging.handlers import TimedRotatingFileHandler
import secrets
import urllib.parse
import time
from datetime import datetime, timezone, timedelta
from flask_sqlalchemy import SQLAlchemy
from functools import wraps
import json
import hashlib

load_dotenv()

# --- Validate environment variables immediately ---
def check_env_vars():
    required = [
        "CLIENT_ID",
        "CLIENT_SECRET",
        "AUTH_BASE_URL",
        "TOKEN_URL",
        "VERIFY_TOKEN",
        "SQLALCHEMY_DATABASE_URI",
        "RATELIMIT_STORAGE_URI",
        "SECRET_KEY",
        "ENVIRONMENT",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    logging.info("All required environment variables are set.")

check_env_vars()

# --- Load variables AFTER validation ---
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AUTH_BASE_URL = os.getenv("AUTH_BASE_URL")
TOKEN_URL = os.getenv("TOKEN_URL")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI")
RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI")

# --- Environment-based settings ---
ENVIRONMENT = os.getenv("ENVIRONMENT", "prod").lower()
DEBUG_MODE = ENVIRONMENT == "dev"

if DEBUG_MODE:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
else:
    os.environ.pop("OAUTHLIB_INSECURE_TRANSPORT", None)

# --- App init ---
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.getenv("SECRET_KEY")

# app.wsgi_app = ProxyFix(app.wsgi_app, x_for=3)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=2, x_proto=2, x_host=2)

os.makedirs('logs', exist_ok=True)

log_file = 'logs/app.log'
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    'logs/app.log',
    when='midnight',       # rotate daily
    backupCount=30,        # keep 30 days
    encoding='utf-8'
)

file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# Load demo data once at startup
DEMO_ACTIVITY_DATA = {}
try:
    with open('static/demo/activity_demo.json', 'r', encoding='utf-8') as f:
        DEMO_ACTIVITY_DATA = json.load(f)
    logger.info("Loaded demo activity data")
except FileNotFoundError:
    logger.warning("Demo data file not found, skipping demo data")
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in demo data file: {e}")

# Session config
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=365)
)

if ENVIRONMENT != "prod":
    app.config.update(
        SESSION_COOKIE_SECURE=False
    )

# DB config
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 2,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
    'max_overflow': 3,
    'pool_timeout': 30
}
db = SQLAlchemy(app)

# --- Models ---
class Athletes(db.Model):
    athlete_id = db.Column(db.BigInteger, primary_key=True)
    access_token = db.Column(db.String(100), nullable=False)
    refresh_token = db.Column(db.String(100), nullable=False)
    expires_at = db.Column(db.BigInteger, nullable=False)

    athlete_username = db.Column(db.String(100))
    athlete_first_name = db.Column(db.String(100))
    athlete_last_name = db.Column(db.String(100))
    athlete_profile = db.Column(db.String(255))
    profile_medium = db.Column(db.String(255))

    first_authentication = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_authentication = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    sex = db.Column(db.String(10))
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))
    country = db.Column(db.String(100))

    follower_count = db.Column(db.Integer)
    friend_count = db.Column(db.Integer)

    date_preference = db.Column(db.String(20))
    measurement_preference = db.Column(db.String(20))

    weight = db.Column(db.Float)
    premium = db.Column(db.Boolean)
    athlete_type = db.Column(db.Integer)

    def __repr__(self):
        return f'<Athletes {self.athlete_id}>'

    def update_from_dict(self, data: dict):
        field_map = {
            'athlete_username': 'username',
            'athlete_first_name': 'firstname',
            'athlete_last_name': 'lastname',
            'athlete_profile': 'profile',
            'profile_medium': 'profile_medium',
            'sex': 'sex',
            'city': 'city',
            'state': 'state',
            'country': 'country',
            'follower_count': 'follower_count',
            'friend_count': 'friend_count',
            'date_preference': 'date_preference',
            'measurement_preference': 'measurement_preference',
            'weight': 'weight',
            'premium': 'premium',
            'athlete_type': 'athlete_type',
        }
        for model_field, data_key in field_map.items():
            setattr(self, model_field, data.get(data_key))

    def update_from_token(self, token_data, athlete_data=None):
        self.access_token = token_data['access_token']
        self.refresh_token = token_data['refresh_token']
        self.expires_at = token_data['expires_at']
        self.last_authentication = datetime.now(timezone.utc)
        if athlete_data:
            self.update_from_dict(athlete_data)

    def update_from_athlete_details(self, details: dict):
        self.update_from_dict(details)

class Activities(db.Model):
    activity_id = db.Column(db.BigInteger, primary_key=True)
    athlete_id = db.Column(db.BigInteger, db.ForeignKey('athletes.athlete_id'), index=True, nullable=False)
    data = db.Column(db.JSON, nullable=False)
    last_synced = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    SYNC_COOLDOWN = timedelta(minutes=3)

    def __repr__(self):
        return f'<Activities {self.athlete_id}:{self.activity_id}>'

    @classmethod
    def get_last_sync(cls, athlete_id, activity_id):
        last_sync = cls.query.filter_by(athlete_id=athlete_id, activity_id=activity_id).order_by(cls.last_synced.desc()).first()
        if last_sync and last_sync.last_synced:
            if last_sync.last_synced.tzinfo is None:
                return last_sync.last_synced.replace(tzinfo=timezone.utc)
            return last_sync.last_synced
        return None
    def is_sync_allowed(self):
        last_sync = self.get_last_sync(self.athlete_id, self.activity_id)
        if not last_sync:
            return True
        current_time = datetime.now(timezone.utc)
        time_since_sync = current_time - last_sync
        return time_since_sync > self.SYNC_COOLDOWN

    def get_cooldown_remaining(self):
        last_sync = self.get_last_sync(self.athlete_id, self.activity_id)
        if not last_sync:
            return 0
        current_time = datetime.now(timezone.utc)
        time_since_sync = current_time - last_sync
        if time_since_sync > self.SYNC_COOLDOWN:
            return 0
        return int((self.SYNC_COOLDOWN - time_since_sync).total_seconds())

class ActivityLists(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    athlete_id = db.Column(db.BigInteger, db.ForeignKey('athletes.athlete_id'), index=True, nullable=False)
    data = db.Column(db.JSON, nullable=False)
    last_synced = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    page = db.Column(db.Integer, nullable=False, default=1)
    per_page = db.Column(db.Integer, nullable=False, default=30)
    SYNC_COOLDOWN = timedelta(minutes=2)
    ITEMS_PER_PAGE = 30

    @classmethod
    def get_last_sync(cls, athlete_id, page=1, per_page=30):
        last_sync = cls.query.filter_by(athlete_id=athlete_id, page=page, per_page=per_page).order_by(cls.last_synced.desc()).first()
        if last_sync and last_sync.last_synced:
            if last_sync.last_synced.tzinfo is None:
                return last_sync.last_synced.replace(tzinfo=timezone.utc)
            return last_sync.last_synced
        return None

    def is_sync_allowed(self):
        last_sync = self.get_last_sync(self.athlete_id, self.page, self.per_page)
        if not last_sync:
            return True
        current_time = datetime.now(timezone.utc)
        time_since_sync = current_time - last_sync
        return time_since_sync > self.SYNC_COOLDOWN

    def get_cooldown_remaining(self):
        last_sync = self.get_last_sync(self.athlete_id, self.page, self.per_page)
        if not last_sync:
            return 0
        current_time = datetime.now(timezone.utc)
        time_since_sync = current_time - last_sync
        if time_since_sync > self.SYNC_COOLDOWN:
            return 0
        return int((self.SYNC_COOLDOWN - time_since_sync).total_seconds())

def fetch_athlete_details(access_token: str) -> dict | None:
    try:
        response = httpx.get(
            "https://www.strava.com/api/v3/athlete",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30.0
        )
        if response.is_success:
            return response.json()
        logger.error(f"Failed to fetch athlete details: {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching athlete details: {e}")
    return None

with app.app_context():
    db.create_all()

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Ensure database sessions are properly cleaned up after each request"""
    db.session.remove()

@app.after_request
def after_request(response: Response) -> Response:
    path = request.path
    
    if path.startswith('/static/'):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'

        if path.endswith(('.css', '.js')):
            response.headers['Cache-Control'] = 'public, max-age=31536000, must-revalidate'
        else:
            response.headers['Cache-Control'] = 'public, max-age=31536000'

        return response
    
    # CSP sources
    csp_base = {
        'default-src': ["'self'"],
        'script-src': ["'self'", "'unsafe-inline'", "www.googletagmanager.com", "www.google-analytics.com", "static.cloudflareinsights.com", "unpkg.com"],
        'style-src': ["'self'", "'unsafe-inline'", "fonts.googleapis.com", "cdnjs.cloudflare.com"],
        'font-src': ["'self'", "fonts.gstatic.com", "cdnjs.cloudflare.com"],
        'img-src': ["'self'", "data:", "*.strava.com", "dgalywyr863hv.cloudfront.net", "www.googletagmanager.com", "www.google-analytics.com"],
        'connect-src': ["'self'", "www.strava.com", "strava.com", "www.google-analytics.com", "region1.google-analytics.com"],
        'frame-ancestors': ["'none'"],
        'form-action': ["'self'"],
        'base-uri': ["'self'"],
    }
    if ENVIRONMENT != "prod":
        csp_base['script-src'].append("cdn.tailwindcss.com")
    
    # Security headers
    response.headers.update({
        'Content-Security-Policy': '; '.join(f"{k} {' '.join(v)}" for k, v in csp_base.items()),
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
        'X-Content-Type-Options': 'nosniff',
        'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Resource-Policy': 'same-origin',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
    })
    
    # Cache-Control (if not already set)
    if 'Cache-Control' not in response.headers:
        if session.get('athlete_id'):
            response.headers['Cache-Control'] = 'private, max-age=600' if path == '/customize' else 'private, no-store'
            if path != '/customize':
                response.headers['Pragma'] = 'no-cache'
        else:
            response.headers['Cache-Control'] = f"public, max-age={300 if path == '/' else 60}"
    
    # Remove server fingerprints
    response.headers.pop('X-Powered-By', None)
    response.headers.pop('Server', None)
    return response

def generate_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']

def validate_csrf_token(request):
    token = request.headers.get('X-CSRF-Token')
    return token and token == session.get('csrf_token')

@app.before_request
def csrf_protect():
    # Protects state-changing requests except exempt endpoints
    if request.path in ['/webhook', '/api/activities/sync']:
        return
    if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
        if not validate_csrf_token(request):
            logger.warning(f'CSRF validation failed - IP: {get_remote_address()}')
            return jsonify({"error": "Invalid CSRF token"}), 403

@app.before_request
def enforce_custom_domain():
    host = request.host.split(":")[0]
    path = request.path
    if path.startswith("/api") or path.startswith("/webhook"):
        return
    allowed_exact = {"fitnessoverlays.com", "localhost", "127.0.0.1"}
    allowed_suffixes = (".ngrok.io", ".ngrok-free.app")
    if host not in allowed_exact and not host.endswith(allowed_suffixes):
        logger.warning(f'Redirecting athlete - IP: {get_remote_address()}')
        return redirect(f"https://fitnessoverlays.com{request.full_path}", code=301)

def refresh_access_token(refresh_token):
    if not refresh_token:
        logger.warning('Missing refresh token')
        return None
    athlete_id = session.get('athlete_id')
    if not athlete_id:
        logger.error('Missing athlete_id in session')
        return None
    athlete = db.session.get(Athletes, athlete_id)
    if not athlete:
        logger.error(f'Athlete {athlete_id} not found')
        return None
    if athlete.refresh_token != refresh_token:
        logger.error('Refresh token mismatch')
        return None
    try:
        response = httpx.post(
            "https://www.strava.com/oauth/token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            },
            timeout=10.0
        )
        response.raise_for_status()
        token_data = response.json()
        
        # Fetch fresh athlete details to keep DB and session in sync
        athlete_details = fetch_athlete_details(token_data['access_token'])
        
        try:
            athlete.update_from_token(token_data, athlete_details)
            db.session.commit()
            logger.info(f'Token refreshed for athlete {athlete_id}')
        except Exception as e:
            db.session.rollback()
            logger.error(f'DB update failed: {str(e)}')
            return None, None

        return token_data, athlete_details
    except httpx.TimeoutException:
        logger.error('Request timed out')
    except httpx.RequestError as e:
        logger.error(f'Network error: {str(e)}')
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
    return None, None

@app.route('/logout', methods=['POST'])
def logout():
    logger.info(f"Athlete logged out - ID: {session.get('athlete_id')} - Name: {session.get('athlete_first_name')} {session.get('athlete_last_name')} - IP: {get_remote_address()}")
    session.clear()
    session['logged_out'] = True
    session.permanent = True
    return redirect('/')

def ensure_valid_token():
    if 'access_token' not in session or 'expires_at' not in session:
        logger.warning("Missing token data in session")
        # Preserve logged_out state if it exists, otherwise clear everything
        was_logged_out = session.get('logged_out')
        session.clear()
        if was_logged_out:
            session['logged_out'] = True
            session.permanent = True
        return False
    current_time = time.time()
    token_expires_at = session['expires_at']
    time_until_expiry = token_expires_at - current_time
    
    # Check if profile data is stale (older than 1 hour)
    last_profile_check = session.get('last_profile_check', 0)
    is_profile_stale = (current_time - last_profile_check) > 3600
    
    if time_until_expiry <= 300 or is_profile_stale:
        if time_until_expiry <= 300:
            logger.info(f"Token expiring in {int(time_until_expiry)}s, attempting refresh")
            new_token, athlete_details = refresh_access_token(session.get('refresh_token'))
            if not new_token:
                logger.warning("Token refresh failed")
                session.clear()
                return False
                
            session['access_token'] = new_token['access_token']
            session['refresh_token'] = new_token['refresh_token']
            session['expires_at'] = new_token['expires_at']
            
            if athlete_details:
                session['athlete_first_name'] = athlete_details.get('firstname')
                session['athlete_last_name'] = athlete_details.get('lastname')
                session['athlete_profile'] = athlete_details.get('profile')
                session['measurement_preference'] = athlete_details.get('measurement_preference')
                session['last_profile_check'] = current_time
                
            logger.info(f"Token refreshed successfully - New expiry: {datetime.fromtimestamp(new_token['expires_at']).isoformat()}")
            return True
            
        elif is_profile_stale:
            # Token is valid, but profile might be stale. Fetch just the profile.
            logger.info("Profile data stale, fetching fresh details")
            athlete_details = fetch_athlete_details(session['access_token'])
            if athlete_details:
                session['athlete_first_name'] = athlete_details.get('firstname')
                session['athlete_last_name'] = athlete_details.get('lastname')
                session['athlete_profile'] = athlete_details.get('profile')
                session['measurement_preference'] = athlete_details.get('measurement_preference')
                session['last_profile_check'] = current_time
                
                # Also update DB
                try:
                    athlete_id = session.get('athlete_id')
                    if athlete_id:
                        athlete = db.session.get(Athletes, athlete_id)
                        if athlete:
                            athlete.update_from_athlete_details(athlete_details)
                            db.session.commit()
                except Exception as e:
                    logger.error(f"Failed to update DB with fresh profile: {e}")
                    
            return True

    logger.info(f"Token valid – expires in {int(time_until_expiry)}s (at {datetime.fromtimestamp(token_expires_at).isoformat()})")
    return True

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'athlete_id' not in session:
            if session.get('logged_out'):
                return render_template('auth_required.html')
            return redirect(url_for('login'))
        athlete = db.session.get(Athletes, session['athlete_id'])
        if not athlete:
            session.clear()
            return redirect(url_for('login'))
        if not ensure_valid_token():
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

CACHED_USER_COUNT = None
LAST_COUNT_UPDATE = None

def get_rounded_user_count():
    global CACHED_USER_COUNT, LAST_COUNT_UPDATE
    
    current_time = datetime.now(timezone.utc)
    
    if CACHED_USER_COUNT is not None and LAST_COUNT_UPDATE:
        if current_time - LAST_COUNT_UPDATE < timedelta(hours=1):
            return CACHED_USER_COUNT

    try:
        count = db.session.query(Athletes).count()
        
        if count < 10:
            rounded = count
        elif count < 100:
            rounded = (count // 10) * 10
        elif count < 1000:
            rounded = (count // 10) * 10
        else:
            rounded = (count // 100) * 100
            
        CACHED_USER_COUNT = rounded
        LAST_COUNT_UPDATE = current_time
        return rounded
    except Exception as e:
        logger.error(f"Error fetching user count: {e}")
        return 200

# --- Rate Limiting ---
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=RATELIMIT_STORAGE_URI,
    default_limits=["200 per day", "50 per hour"],
    default_limits_exempt_when=lambda: request.path.startswith('/static/') or request.path in ['/favicon.ico', '/robots.txt', '/llms.txt', '/sitemap.xml']
)

@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(f"Rate limit exceeded - IP: {get_remote_address()} - Path: {request.path}")
    if request.path.startswith('/api/') or request.path == '/webhook':
        return jsonify({
            "error": "Rate limit exceeded",
            "message": str(e.description)
        }), 429
    return redirect(url_for('index'))
limiter.error_handler = ratelimit_handler

@app.route('/login')
@limiter.limit("5 per minute")
def login():
    try:
        # Clear any logged_out flag on manual login attempt
        session.pop('logged_out', None)
            
        callback_url = url_for('callback', _external=True)
        logger.info(f"Generated callback URL: {callback_url}")
        
        # Manual OAuth2 URL construction
        state = secrets.token_urlsafe(16)
        session['oauth_state'] = state
        
        params = {
            "client_id": CLIENT_ID,
            "redirect_uri": callback_url,
            "response_type": "code",
            "scope": "read,activity:read_all,profile:read_all",
            "state": state
        }
        authorization_url = f"{AUTH_BASE_URL}?{urllib.parse.urlencode(params)}"
        
        return redirect(authorization_url)
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Authentication failed"}), 401

@app.route('/callback')
@limiter.limit("5 per minute")
def callback():
    try:
        if 'error' in request.args:
            logger.info(f"User denied Strava authorization - IP: {get_remote_address()}")
            return redirect('/')
            
        # Verify state
        state = request.args.get('state')
        if not state or state != session.get('oauth_state'):
             logger.error(f"Invalid state parameter - IP: {get_remote_address()}")
             return redirect('/')

        code = request.args.get('code')
        if not code:
            logger.error(f"Missing code parameter - IP: {get_remote_address()}")
            return redirect('/')

        callback_url = url_for('callback', _external=True)
        logger.info(f"Using dynamic callback URL in callback handler: {callback_url}") 
        
        # Exchange code for token using httpx
        try:
            token_response = httpx.post(
                TOKEN_URL,
                data={
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": callback_url
                },
                timeout=10.0
            )
            token_response.raise_for_status()
            token = token_response.json()
        except httpx.HTTPError as e:
             logger.error(f"Token exchange failed: {e}")
             return redirect('/')

        athlete_data = token.get('athlete', {})
        athlete_id = athlete_data.get('id')
        logger.info(f"Athlete Data: {athlete_data}")
        if not athlete_id:
            logger.error('No athlete ID in token response')
            return redirect('/')
        logger.info(f"Athlete Logged In - ID: {athlete_id} - Name: {athlete_data.get('firstname')} {athlete_data.get('lastname')} - IP: {get_remote_address()}")
        try:
            athlete = db.session.get(Athletes, athlete_id)
            if not athlete:
                athlete = Athletes(athlete_id=athlete_id)
                db.session.add(athlete)
                logger.info(f'Creating new athlete record for ID: {athlete_id}')
            else:
                logger.info(f'Updating existing athlete record for ID: {athlete_id}')
            athlete.update_from_token(token, athlete_data)
            athlete_details = fetch_athlete_details(token['access_token'])
            if athlete_details:
                athlete.update_from_athlete_details(athlete_details)
            db.session.commit()
            logger.info(f'Successfully updated athlete data in database for ID: {athlete_id}')
            session.permanent = True
            session['athlete_id'] = athlete_id
            session['athlete_username'] = athlete_data.get('username')
            session['athlete_first_name'] = athlete_data.get('firstname')
            session['athlete_last_name'] = athlete_data.get('lastname')
            session['athlete_profile'] = athlete_data.get('profile_medium')
            session['access_token'] = token['access_token']
            session['refresh_token'] = token['refresh_token']
            session['expires_at'] = token['expires_at']
            session['date_preference'] = athlete_data.get('date_preference')
            session['measurement_preference'] = athlete_data.get('measurement_preference')
            session['csrf_token'] = generate_csrf_token()
        except Exception as db_error:
            db.session.rollback()
            logger.error(f'Database error during callback: {str(db_error)}')
            return redirect('/')
        return redirect('/')
    except Exception as e:
        logger.error(f'OAuth callback error details: {str(e)} - IP: {get_remote_address()}')
        return redirect('/')

@limiter.limit("100 per hour", key_func=lambda: session.get("athlete_id", get_remote_address()))
@app.route('/customize')
@login_required
def customize():
    logger.info(f"Customize overlays endpoint called - Athlete ID: {session['athlete_id']} - Athlete Name: {session['athlete_first_name']} {session['athlete_last_name']}")
    return render_template("customize.html",
                        authenticated=True,
                        athlete_id=session.get("athlete_id"),
                        athlete_first_name=session.get("athlete_first_name"),
                        athlete_last_name=session.get("athlete_last_name"),
                        athlete_profile=session.get("athlete_profile"),
                        measurement_preference=session.get("measurement_preference"),
                        csrf_token=session['csrf_token'])

@limiter.limit("100 per hour", key_func=lambda: session.get("athlete_id", get_remote_address()))
@app.route('/activities')
@login_required
def activities():
    return render_template("activities.html",
                        authenticated=True,
                        athlete_id=session.get("athlete_id"),
                        athlete_first_name=session.get("athlete_first_name"),
                        athlete_last_name=session.get("athlete_last_name"),
                        athlete_profile=session.get("athlete_profile"),
                        measurement_preference=session.get("measurement_preference"),
                        csrf_token=session['csrf_token'])

def create_sync_response(activities, page, per_page, sync_log, seconds_remaining, warning=None, using_cached=False):
    response = {
        "activities": activities,
        "pagination": {"page": page, "per_page": per_page},
        "cooldown": {
            "active": seconds_remaining > 0,
            "seconds_remaining": seconds_remaining,
            "total_cooldown": ActivityLists.SYNC_COOLDOWN.total_seconds()
        },
        "cached": using_cached
    }
    if sync_log and sync_log.last_synced:
        response["last_synced"] = sync_log.last_synced.isoformat()
    elif not sync_log:
        response["last_synced"] = datetime.now(timezone.utc).isoformat()
    if warning:
        response["warning"] = warning
    return response

@limiter.limit("100 per hour", key_func=lambda: session.get("athlete_id", get_remote_address()))
@app.route('/api/activities/sync', methods=['GET'])
@login_required
def sync_activities():
    logger.info(f"Syncing Activities called - Athlete ID: {session['athlete_id']} - Athlete Name: {session['athlete_first_name']} {session['athlete_last_name']}")
    athlete_id = session.get('athlete_id')
    if not athlete_id:
        return jsonify({"error": "Not authenticated"}), 401
    page = max(1, request.args.get('page', 1, type=int))
    per_page = min(request.args.get('per_page', 30, type=int), ActivityLists.ITEMS_PER_PAGE)
    sync_instance = ActivityLists(athlete_id=athlete_id, page=page, per_page=per_page)
    sync_log = ActivityLists.query.filter_by(
        athlete_id=athlete_id,
        page=page,
        per_page=per_page
    ).first()
    seconds_remaining = sync_instance.get_cooldown_remaining()
    
    # Generate ETag from last_synced timestamp and athlete_id
    if sync_log and sync_log.last_synced:
        etag_source = f"{athlete_id}:{page}:{per_page}:{sync_log.last_synced.isoformat()}"
        etag = hashlib.md5(etag_source.encode()).hexdigest()
    else:
        etag = None
    
    # Only return 304 if sync is NOT allowed (cooldown active) AND ETag matches
    if not sync_instance.is_sync_allowed() and sync_log:
        # Check If-None-Match header for conditional request
        if etag and request.headers.get('If-None-Match') == etag:
            response = make_response('', 304)
            response.headers['ETag'] = etag
            response.headers['Cache-Control'] = 'private, max-age=180'
            logger.info(f"Returning 304 Not Modified for activities sync (Cooldown Active) - Athlete ID: {athlete_id}")
            return response
    
    if not sync_instance.is_sync_allowed() and sync_log:
        response_data = create_sync_response(sync_log.data if sync_log else [], page, per_page, sync_log, seconds_remaining, using_cached=True)
        response = make_response(jsonify(response_data))
        if sync_log and sync_log.last_synced:
            etag_source = f"{athlete_id}:{page}:{per_page}:{sync_log.last_synced.isoformat()}"
            etag = hashlib.md5(etag_source.encode()).hexdigest()
            response.headers['ETag'] = etag
        response.headers['Cache-Control'] = 'private, max-age=180'
        return response
        
    try:
        response = httpx.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers={"Authorization": f"Bearer {session['access_token']}"},
            params={"page": page, "per_page": per_page},
            timeout=15.0
        )
        if not response.is_success:
            response_data = create_sync_response(
                sync_log.data if sync_log else [],
                page,
                per_page,
                sync_log,
                seconds_remaining,
                warning="Failed to fetch fresh data, showing cached data" if sync_log else None,
                using_cached=True
            )
            flask_response = make_response(jsonify(response_data), response.status_code if not sync_log else 200)
            if sync_log and sync_log.last_synced:
                etag_source = f"{athlete_id}:{page}:{per_page}:{sync_log.last_synced.isoformat()}"
                etag = hashlib.md5(etag_source.encode()).hexdigest()
                flask_response.headers['ETag'] = etag
            flask_response.headers['Cache-Control'] = 'private, max-age=180'
            return flask_response
            
        activities = response.json()
        current_time = datetime.now(timezone.utc)
        try:
            if sync_log:
                sync_log.data = activities
                sync_log.last_synced = current_time
            else:
                sync_log = ActivityLists(
                    athlete_id=athlete_id,
                    data=activities,
                    page=page,
                    per_page=per_page,
                    last_synced=current_time
                )
                db.session.add(sync_log)
            db.session.commit()
            
            response_data = create_sync_response(activities, page, per_page, sync_log, seconds_remaining, using_cached=False)
            flask_response = make_response(jsonify(response_data))
            
            # Generate fresh ETag
            etag_source = f"{athlete_id}:{page}:{per_page}:{current_time.isoformat()}"
            etag = hashlib.md5(etag_source.encode()).hexdigest()
            flask_response.headers['ETag'] = etag
            flask_response.headers['Cache-Control'] = 'private, max-age=180'
            
            return flask_response
        except Exception as db_error:
            db.session.rollback()
            logger.error(f"Database error during sync: {db_error}")
            response_data = create_sync_response(activities, page, per_page, sync_log, seconds_remaining, using_cached=False)
            flask_response = make_response(jsonify(response_data))
            flask_response.headers['Cache-Control'] = 'private, max-age=180'
            return flask_response
    except Exception as e:
        logger.error(f"Sync error: {e}")
        response_data = create_sync_response(
            sync_log.data if sync_log else [],
            page,
            per_page,
            sync_log,
            seconds_remaining,
            warning="Failed to sync activities",
            using_cached=True
        )
        flask_response = make_response(jsonify(response_data), 500)
        if sync_log and sync_log.last_synced:
            etag_source = f"{athlete_id}:{page}:{per_page}:{sync_log.last_synced.isoformat()}"
            etag = hashlib.md5(etag_source.encode()).hexdigest()
            flask_response.headers['ETag'] = etag
        flask_response.headers['Cache-Control'] = 'private, max-age=180'
        return flask_response

@app.route('/')
def index():
    logger.info(f"Landing page accessed - IP: {get_remote_address()} | X-Forwarded-For: {request.headers.get('X-Forwarded-For')}")
    try:
        csrf_token = generate_csrf_token()
        user_count = get_rounded_user_count()
        
        if "access_token" not in session or not ensure_valid_token():
            logger.info("Index: No access token in session or invalid token")
            return render_template("index.html", 
                                    authenticated=False, 
                                    csrf_token=csrf_token,
                                    user_count=user_count)
        logger.info(f"Index: Authenticated user - Athlete ID: {session.get('athlete_id')} - Athlete Name: {session['athlete_first_name']} {session['athlete_last_name']}")
        return render_template("index.html",
                                authenticated=True,
                                athlete_id=session.get("athlete_id"),
                                athlete_first_name=session.get("athlete_first_name"),
                                athlete_last_name=session.get("athlete_last_name"),
                                athlete_profile=session.get("athlete_profile"),
                                csrf_token=csrf_token,
                                user_count=user_count)
    except Exception as e:
        logger.error(f"Index: Unexpected error: {str(e)}")
        session.clear()
        return render_template("index.html", 
                                authenticated=False, 
                                csrf_token=generate_csrf_token(),
                                user_count=200)

@limiter.limit("100 per hour", key_func=lambda: session.get("athlete_id", get_remote_address()))
@app.route('/api/activities/<int:activity_id>', methods=['GET'])
@login_required
def get_activity(activity_id):
    athlete_id = session['athlete_id']
    logger.info(f"Fetching Activity {activity_id} - Athlete ID: {athlete_id}")
    activity = Activities.query.filter_by(
        athlete_id=athlete_id,
        activity_id=activity_id
    ).first()
    
    # Generate ETag from last_synced timestamp and activity_id
    if activity and activity.last_synced:
        etag_source = f"{athlete_id}:{activity_id}:{activity.last_synced.isoformat()}"
        etag = hashlib.md5(etag_source.encode()).hexdigest()
    else:
        etag = None
    
    if activity:
        seconds_remaining = activity.get_cooldown_remaining()
        if seconds_remaining > 0:
            # Only return 304 if cooldown is active AND ETag matches
            if etag and request.headers.get('If-None-Match') == etag:
                response = make_response('', 304)
                response.headers['ETag'] = etag
                response.headers['Cache-Control'] = 'private, max-age=300'
                logger.info(f"Returning 304 Not Modified for activity {activity_id} (Cooldown Active) - Athlete ID: {athlete_id}")
                return response

            logger.info(f"Returning cached activity {activity_id} - {seconds_remaining}s cooldown remaining")
            response_data = {
                "activity": activity.data,
                "cooldown": {
                    "active": True,
                    "seconds_remaining": seconds_remaining,
                    "total_cooldown": Activities.SYNC_COOLDOWN.total_seconds()
                },
                "cached": True
            }
            flask_response = make_response(jsonify(response_data))
            if etag:
                flask_response.headers['ETag'] = etag
            flask_response.headers['Cache-Control'] = 'private, max-age=300'
            return flask_response
            
    try:
        logger.info(f"Fetching fresh data for activity {activity_id} from Strava")
        response = httpx.get(
            f"https://www.strava.com/api/v3/activities/{activity_id}",
            headers={"Authorization": f"Bearer {session['access_token']}"},
            timeout=15.0
        )
        if not response.is_success:
            logger.error(f"Strava API error for activity {activity_id}: {response.status_code}")
            if activity:
                seconds_remaining = activity.get_cooldown_remaining()
                response_data = {
                    "activity": activity.data,
                    "cooldown": {
                        "active": True,
                        "seconds_remaining": seconds_remaining,
                        "total_cooldown": Activities.SYNC_COOLDOWN.total_seconds()
                    },
                    "cached": True,
                    "warning": "Failed to fetch fresh data, showing cached data"
                }
                flask_response = make_response(jsonify(response_data), response.status_code)
                if activity.last_synced:
                    etag_source = f"{athlete_id}:{activity_id}:{activity.last_synced.isoformat()}"
                    etag = hashlib.md5(etag_source.encode()).hexdigest()
                    flask_response.headers['ETag'] = etag
                flask_response.headers['Cache-Control'] = 'private, max-age=300'
                return flask_response
            return jsonify({"error": "Failed to fetch activity data"}), response.status_code
            
        activity_data = response.json()
        current_time = datetime.now(timezone.utc)
        try:
            if activity:
                activity.data = activity_data
                activity.last_synced = current_time
            else:
                activity = Activities(
                    activity_id=activity_id,
                    athlete_id=athlete_id,
                    data=activity_data,
                    last_synced=current_time
                )
                db.session.add(activity)
            db.session.commit()
            logger.info(f"Successfully cached activity {activity_id}")
            
            response_data = {
                "activity": activity_data,
                "cooldown": {
                    "active": False,
                    "seconds_remaining": 0,
                    "total_cooldown": Activities.SYNC_COOLDOWN.total_seconds()
                },
                "cached": False
            }
            flask_response = make_response(jsonify(response_data))
            
            # Generate fresh ETag
            etag_source = f"{athlete_id}:{activity_id}:{current_time.isoformat()}"
            etag = hashlib.md5(etag_source.encode()).hexdigest()
            flask_response.headers['ETag'] = etag
            flask_response.headers['Cache-Control'] = 'private, max-age=300'
            
            return flask_response
        except Exception as db_error:
            db.session.rollback()
            logger.error(f'Database error caching activity {activity_id}: {str(db_error)}')
            raise
    except Exception as e:
        logger.error(f'Error fetching activity {activity_id}: {str(e)}')
        if activity:
            seconds_remaining = activity.get_cooldown_remaining()
            response_data = {
                "activity": activity.data,
                "cooldown": {
                    "active": True,
                    "seconds_remaining": seconds_remaining,
                    "total_cooldown": Activities.SYNC_COOLDOWN.total_seconds()
                },
                "cached": True,
                "warning": "Failed to fetch fresh data, showing cached data"
            }
            flask_response = make_response(jsonify(response_data), 500)
            if activity.last_synced:
                etag_source = f"{athlete_id}:{activity_id}:{activity.last_synced.isoformat()}"
                etag = hashlib.md5(etag_source.encode()).hexdigest()
                flask_response.headers['ETag'] = etag
            flask_response.headers['Cache-Control'] = 'private, max-age=300'
            return flask_response
        return jsonify({"error": "Failed to fetch activity data"}), 500

def delete_user_data(athlete_id):
    try:
        logger.info(f"Deleting data for athlete {athlete_id}")
        Activities.query.filter_by(athlete_id=athlete_id).delete()
        ActivityLists.query.filter_by(athlete_id=athlete_id).delete()
        Athletes.query.filter_by(athlete_id=athlete_id).delete()
        db.session.commit()
        logger.info(f"Successfully deleted all data for athlete {athlete_id}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting user data for athlete {athlete_id}: {str(e)}")
        raise

@app.route("/webhook", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def webhook():
    # Handles Strava webhook verification and deauthorization events
    if request.method == "GET":
        if not VERIFY_TOKEN:
            logger.error("STRAVA_VERIFY_TOKEN not configured")
            return jsonify({"error": "Webhook not configured"}), 500
        if request.args.get("hub.verify_token") == VERIFY_TOKEN:
            challenge = request.args.get("hub.challenge")
            logger.info(f"Webhook verification successful with challenge: {challenge}")
            return jsonify({
                "hub.challenge": challenge
            })
        logger.warning(f"Invalid webhook verification token from IP: {get_remote_address()}")
        return jsonify({"error": "Invalid verification token"}), 403
    elif request.method == "POST":
        try:
            signature = request.headers.get('X-Strava-Signature')
            if not signature:
                if ENVIRONMENT == "dev":
                    logger.warning("Skipping signature verification in development mode")
                else:
                    logger.warning(f"Missing Strava signature in webhook request from IP: {get_remote_address()}")
                    return jsonify({"error": "Unauthorized"}), 403
            event = request.json
            logger.info(f"Received webhook event: {event}")
            if (event.get("object_type") == "athlete" and
                event.get("aspect_type") == "update" and
                event.get("updates", {}).get("authorized") == "false"):
                athlete_id = event.get("owner_id")
                if athlete_id:
                    delete_user_data(athlete_id)
                    logger.info(f"Successfully processed deauthorization for athlete {athlete_id}")
                else:
                    logger.warning("Received deauthorization event without athlete_id")
            return jsonify({"status": "ok"}), 200
        except Exception as e:
            logger.error(f"Error processing webhook event: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route("/favicon.ico")
def favicon():
    return redirect(url_for('static', filename='images/FitnessOverlaysLogo.ico'), code=302)

@app.route('/robots.txt')
def robots_txt():
    return redirect(url_for('static', filename='robots.txt'), code=302)

@app.route('/llms.txt')
def llms_txt():
    return redirect(url_for('static', filename='llms.txt'), code=302)

@limiter.limit("30 per hour")
@app.route('/demo')
def demo():
    logger.info(f"Demo page accessed - IP: {get_remote_address()}")
    activity = DEMO_ACTIVITY_DATA
    is_authenticated = False
    athlete_id = session.get("athlete_id")
    if athlete_id:
        athlete = db.session.get(Athletes, athlete_id)
        if athlete:
            is_authenticated = True
            logger.info(f"Authenticated user accessing demo - Athlete ID: {athlete_id} - Name: {session.get('athlete_first_name')} {session.get('athlete_last_name')} - IP: {get_remote_address()}")
    return render_template(
        'customize.html',
        activity=activity,
        demo_mode=True,
        authenticated=is_authenticated,
        athlete_id=session.get("athlete_id") if is_authenticated else None,
        athlete_first_name=session.get("athlete_first_name") if is_authenticated else None,
        athlete_last_name=session.get("athlete_last_name") if is_authenticated else None,
        athlete_profile=session.get("athlete_profile") if is_authenticated else None,
        csrf_token=session.get('csrf_token') if is_authenticated else generate_csrf_token()
    )

def get_lastmod(filepath):
    try:
        timestamp = os.path.getmtime(filepath)
        if filepath.endswith('.html'):
            base_timestamp = os.path.getmtime('templates/base.html')
            timestamp = max(timestamp, base_timestamp)
        dt = datetime.fromtimestamp(timestamp, timezone.utc)
        return dt.date().isoformat()
    except Exception:
        return None

@app.route('/sitemap.xml')
def sitemap_xml():
    sitemap = f'''<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url>
            <loc>https://fitnessoverlays.com/</loc>
            {f'<lastmod>{get_lastmod("templates/index.html")}</lastmod>' if get_lastmod("templates/index.html") else ''}
            <changefreq>weekly</changefreq>
            <priority>1.0</priority>
        </url>
        <url>
            <loc>https://fitnessoverlays.com/demo</loc>
            {f'<lastmod>{get_lastmod("templates/customize.html")}</lastmod>' if get_lastmod("templates/customize.html") else ''}
            <changefreq>weekly</changefreq>
            <priority>1.0</priority>
        </url>
    </urlset>'''
    return Response(sitemap, mimetype='application/xml')

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    app.run(debug=DEBUG_MODE, port=8000)
