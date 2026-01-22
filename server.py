from fastapi import FastAPI, Request, Response, Depends, HTTPException, Query, Form
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette_session import SessionMiddleware
from starlette_session.backends import BackendType
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import BigInteger, String, Boolean, Integer, Float, DateTime, ForeignKey, JSON, select, delete, func
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional
import os
import httpx
import logging
from logging.handlers import TimedRotatingFileHandler
import secrets
import urllib.parse
import time
import json
import hashlib

# Environment setup
load_dotenv()

def check_env_vars():
    required = [
        "CLIENT_ID", "CLIENT_SECRET", "AUTH_BASE_URL", "TOKEN_URL", "VERIFY_TOKEN",
        "SQLALCHEMY_DATABASE_URI", "RATELIMIT_STORAGE_URI", "SECRET_KEY", "ENVIRONMENT",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

check_env_vars()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AUTH_BASE_URL = os.getenv("AUTH_BASE_URL")
TOKEN_URL = os.getenv("TOKEN_URL")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI")
RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI")
SECRET_KEY = os.getenv("SECRET_KEY")

ENVIRONMENT = os.getenv("ENVIRONMENT", "prod").lower()
DEBUG_MODE = ENVIRONMENT == "dev"
USE_REDIS = RATELIMIT_STORAGE_URI.startswith("redis://")


# Logging setup
os.makedirs('logs', exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s')

file_handler = TimedRotatingFileHandler('logs/app.log', when='midnight', backupCount=30, encoding='utf-8')
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# Silence noisy dependency loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

SESSION_COOKIE_NAME = "session"
SESSION_MAX_AGE = 60 * 60 * 24 * 365 

# Database setup
# Using asyncpg driver with connection pooling enabled
async_db_url = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql+asyncpg://").replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(
    async_db_url,
    pool_size=2,           # Keep 2 connections open
    pool_recycle=3600,     # Recycle connections every hour
    pool_pre_ping=True,    # Test connection before use
    max_overflow=3,        # Allow 3 extra connections during spikes
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

# --- Models ---
class Athletes(Base):
    __tablename__ = "athletes"
    
    athlete_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    access_token: Mapped[str] = mapped_column(String(100), nullable=False)
    refresh_token: Mapped[str] = mapped_column(String(100), nullable=False)
    expires_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    athlete_username: Mapped[Optional[str]] = mapped_column(String(100))
    athlete_first_name: Mapped[Optional[str]] = mapped_column(String(100))
    athlete_last_name: Mapped[Optional[str]] = mapped_column(String(100))
    athlete_profile: Mapped[Optional[str]] = mapped_column(String(255))
    profile_medium: Mapped[Optional[str]] = mapped_column(String(255))

    first_authentication: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_authentication: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    sex: Mapped[Optional[str]] = mapped_column(String(10))
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[Optional[str]] = mapped_column(String(100))

    follower_count: Mapped[Optional[int]] = mapped_column(Integer)
    friend_count: Mapped[Optional[int]] = mapped_column(Integer)

    date_preference: Mapped[Optional[str]] = mapped_column(String(20))
    measurement_preference: Mapped[Optional[str]] = mapped_column(String(20))

    weight: Mapped[Optional[float]] = mapped_column(Float)
    premium: Mapped[Optional[bool]] = mapped_column(Boolean)
    athlete_type: Mapped[Optional[int]] = mapped_column(Integer)

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


class Activities(Base):
    __tablename__ = "activities"
    
    activity_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(BigInteger, ForeignKey('athletes.athlete_id'), index=True, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    last_synced: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    SYNC_COOLDOWN = timedelta(minutes=3)

    def __repr__(self):
        return f'<Activities {self.athlete_id}:{self.activity_id}>'

    @classmethod
    async def get_last_sync(cls, db: AsyncSession, athlete_id: int, activity_id: int) -> Optional[datetime]:
        result = await db.execute(
            select(cls).where(cls.athlete_id == athlete_id, cls.activity_id == activity_id).order_by(cls.last_synced.desc())
        )
        last_sync = result.scalar_one_or_none()
        if last_sync and last_sync.last_synced:
            if last_sync.last_synced.tzinfo is None:
                return last_sync.last_synced.replace(tzinfo=timezone.utc)
            return last_sync.last_synced
        return None

    async def is_sync_allowed(self, db: AsyncSession) -> bool:
        last_sync = await self.get_last_sync(db, self.athlete_id, self.activity_id)
        if not last_sync:
            return True
        current_time = datetime.now(timezone.utc)
        time_since_sync = current_time - last_sync
        return time_since_sync > self.SYNC_COOLDOWN

    async def get_cooldown_remaining(self, db: AsyncSession) -> int:
        last_sync = await self.get_last_sync(db, self.athlete_id, self.activity_id)
        if not last_sync:
            return 0
        current_time = datetime.now(timezone.utc)
        time_since_sync = current_time - last_sync
        if time_since_sync > self.SYNC_COOLDOWN:
            return 0
        return int((self.SYNC_COOLDOWN - time_since_sync).total_seconds())


class ActivityLists(Base):
    __tablename__ = "activity_lists"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    athlete_id: Mapped[int] = mapped_column(BigInteger, ForeignKey('athletes.athlete_id'), index=True, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    last_synced: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    page: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    per_page: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    SYNC_COOLDOWN = timedelta(minutes=2)
    ITEMS_PER_PAGE = 30

    @classmethod
    async def get_last_sync(cls, db: AsyncSession, athlete_id: int, page: int = 1, per_page: int = 30) -> Optional[datetime]:
        result = await db.execute(
            select(cls).where(cls.athlete_id == athlete_id, cls.page == page, cls.per_page == per_page).order_by(cls.last_synced.desc())
        )
        last_sync = result.scalar_one_or_none()
        if last_sync and last_sync.last_synced:
            if last_sync.last_synced.tzinfo is None:
                return last_sync.last_synced.replace(tzinfo=timezone.utc)
            return last_sync.last_synced
        return None

    async def is_sync_allowed(self, db: AsyncSession) -> bool:
        last_sync = await self.get_last_sync(db, self.athlete_id, self.page, self.per_page)
        if not last_sync:
            return True
        current_time = datetime.now(timezone.utc)
        time_since_sync = current_time - last_sync
        return time_since_sync > self.SYNC_COOLDOWN

    async def get_cooldown_remaining(self, db: AsyncSession) -> int:
        last_sync = await self.get_last_sync(db, self.athlete_id, self.page, self.per_page)
        if not last_sync:
            return 0
        current_time = datetime.now(timezone.utc)
        time_since_sync = current_time - last_sync
        if time_since_sync > self.SYNC_COOLDOWN:
            return 0
        return int((self.SYNC_COOLDOWN - time_since_sync).total_seconds())


# --- Load demo data ---
DEMO_ACTIVITY_DATA = {}
def load_demo_data():
    global DEMO_ACTIVITY_DATA
    try:
        with open('static/demo/activity_demo.json', 'r', encoding='utf-8') as f:
            DEMO_ACTIVITY_DATA = json.load(f)
        return True
    except Exception:
        return False


# --- User count cache ---
CACHED_USER_COUNT = None
LAST_COUNT_UPDATE = None

async def get_rounded_user_count(db: AsyncSession) -> int:
    global CACHED_USER_COUNT, LAST_COUNT_UPDATE
    
    current_time = datetime.now(timezone.utc)
    
    if CACHED_USER_COUNT is not None and LAST_COUNT_UPDATE:
        if current_time - LAST_COUNT_UPDATE < timedelta(hours=1):
            return CACHED_USER_COUNT

    try:
        result = await db.execute(select(func.count()).select_from(Athletes))
        count = result.scalar()
        
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


# --- HTTP client ---
http_client: httpx.AsyncClient = None

async def get_http_client() -> httpx.AsyncClient:
    return http_client


async def fetch_athlete_details(access_token: str) -> Optional[dict]:
    try:
        response = await http_client.get(
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


# --- Custom IP extraction for proxy chain (Cloudflare -> Render) ---
def get_real_ip(request: Request) -> str:
    """Get client IP from Cloudflare header, X-Real-IP, X-Forwarded-For, or direct."""
    return (
        request.headers.get("CF-Connecting-IP") or
        request.headers.get("X-Real-IP") or
        (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip() or
        (request.client.host if request.client else "unknown")
    )


# --- Rate limiter with custom key function ---
limiter = Limiter(key_func=get_real_ip, storage_uri=RATELIMIT_STORAGE_URI)


# --- App lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    http_client = httpx.AsyncClient()
    
    if load_demo_data():
        logger.info("Loaded demo activity data")
    else:
        logger.warning("Demo data file not found or invalid")

    # Silence noisy loggers again at startup to be sure
    for noisy in ["uvicorn.access", "httpx", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    session_type = "Redis" if USE_REDIS else "Cookie"
    logger.info(f"FastAPI started - Session Storage: {session_type} - Debug: {DEBUG_MODE}")

    yield
    # Shutdown
    await http_client.aclose()
    await engine.dispose()
    logger.info("FastAPI application shutdown")


# --- App init ---
app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.state.limiter = limiter


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


# --- CSRF helpers ---
def generate_csrf_token(request: Request) -> str:
    """Generate or retrieve CSRF token from session"""
    if 'csrf_token' not in request.session:
        request.session['csrf_token'] = secrets.token_hex(32)
    return request.session['csrf_token']


def validate_csrf_token(request: Request) -> bool:
    """Validate CSRF token from header against session"""
    token = request.headers.get('X-CSRF-Token')
    return token and token == request.session.get('csrf_token')


# --- Database dependency ---
async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


# --- Security headers middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        
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
        
        response.headers['Content-Security-Policy'] = '; '.join(f"{k} {' '.join(v)}" for k, v in csp_base.items())
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Remove server fingerprints
        if 'server' in response.headers:
            del response.headers['server']
        
        return response


# --- CSRF middleware ---
class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Exempt endpoints
        if request.url.path in ['/webhook', '/api/activities/sync']:
            return await call_next(request)
        
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            if not validate_csrf_token(request):
                logger.warning(f'[{get_real_ip(request)}] CSRF validation failed')
                return JSONResponse({"error": "Invalid CSRF token"}, status_code=403)
        
        return await call_next(request)


# --- Domain enforcement middleware ---
class DomainEnforcementMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "").split(":")[0]
        path = request.url.path
        
        if path.startswith("/api") or path.startswith("/webhook"):
            return await call_next(request)
        
        allowed_exact = {"fitnessoverlays.com", "localhost", "127.0.0.1"}
        allowed_suffixes = (".ngrok.io", ".ngrok-free.app")
        
        if host not in allowed_exact and not any(host.endswith(suffix) for suffix in allowed_suffixes):
            logger.warning(f'[{get_real_ip(request)}] Redirecting athlete to canonical domain')
            return RedirectResponse(f"https://fitnessoverlays.com{request.url.path}", status_code=301)
        
        return await call_next(request)


# Add middlewares (order matters - last added is executed first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CSRFMiddleware)
app.add_middleware(DomainEnforcementMiddleware)


if USE_REDIS:
    # Production: Redis backend
    import redis
    redis_client = redis.from_url(RATELIMIT_STORAGE_URI)
    app.add_middleware(
        SessionMiddleware,
        secret_key=SECRET_KEY,
        cookie_name=SESSION_COOKIE_NAME,
        max_age=SESSION_MAX_AGE,
        same_site="lax",
        https_only=True,
        backend_type=BackendType.redis,
        backend_client=redis_client,
    )
else:
    # Development: Cookie-based sessions
    app.add_middleware(
        SessionMiddleware,
        secret_key=SECRET_KEY,
        cookie_name=SESSION_COOKIE_NAME,
        max_age=SESSION_MAX_AGE,
        same_site="lax",
        https_only=False,
        backend_type=BackendType.cookie,
    )


# --- Rate limit handler ---
@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(f"[{get_real_ip(request)}] Rate limit exceeded - Path: {request.url.path} - Info: {str(exc.detail)}")
    if request.url.path.startswith('/api/') or request.url.path == '/webhook':
        return JSONResponse(
            {"error": "Rate limit exceeded", "message": str(exc.detail)},
            status_code=429
        )
    return RedirectResponse("/", status_code=303)


# --- Auth helpers ---
async def refresh_access_token(request: Request, db: AsyncSession) -> tuple[Optional[dict], Optional[dict]]:
    """Refresh access token using refresh token from session"""
    refresh_token = request.session.get('refresh_token')
    if not refresh_token:
        logger.warning(f"[{get_real_ip(request)}] Refresh token missing in session")
        return None, None
    
    athlete_id = request.session.get('athlete_id')
    if not athlete_id:
        logger.error(f"[{get_real_ip(request)}] athlete_id missing in session during refresh")
        return None, None
    
    result = await db.execute(select(Athletes).where(Athletes.athlete_id == athlete_id))
    athlete = result.scalar_one_or_none()
    
    if not athlete:
        logger.error(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Athlete record not found during refresh")
        return None, None
    
    if athlete.refresh_token != refresh_token:
        logger.error(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Refresh token mismatch")
        return None, None
    
    try:
        response = await http_client.post(
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
        
        athlete_details = await fetch_athlete_details(token_data['access_token'])
        
        try:
            athlete.update_from_token(token_data, athlete_details)
            await db.commit()
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Token refreshed successfully")
        except Exception as e:
            await db.rollback()
            logger.error(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] DB update failed during refresh: {str(e)}")
            return None, None

        return token_data, athlete_details
    except httpx.TimeoutException:
        logger.error(f"[{get_real_ip(request)}] Refresh request timed out")
    except httpx.RequestError as e:
        logger.error(f"[{get_real_ip(request)}] Refresh network error: {str(e)}")
    except Exception as e:
        logger.error(f"[{get_real_ip(request)}] Unexpected error during refresh: {str(e)}")
    return None, None


async def ensure_valid_token(request: Request, db: AsyncSession) -> bool:
    """Ensure token is valid, refresh if needed. Updates session directly. Returns True if valid."""
    if 'access_token' not in request.session or 'expires_at' not in request.session:
        logger.warning(f"[{get_real_ip(request)}] Token missing in session, clearing")
        was_logged_out = request.session.get('logged_out')
        request.session.clear()
        if was_logged_out:
            request.session['logged_out'] = True
        return False
    
    current_time = time.time()
    athlete_id = request.session.get('athlete_id', 'Unknown')
    token_expires_at = request.session['expires_at']
    time_until_expiry = token_expires_at - current_time
    
    last_profile_check = request.session.get('last_profile_check', 0)
    is_profile_stale = (current_time - last_profile_check) > 3600
    
    if time_until_expiry <= 300 or is_profile_stale:
        if time_until_expiry <= 300:
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Token expiring in {int(time_until_expiry)}s, attempting refresh")
            new_token, athlete_details = await refresh_access_token(request, db)
            if not new_token:
                logger.warning(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Token refresh failed")
                request.session.clear()
                return False
            
            request.session['access_token'] = new_token['access_token']
            request.session['refresh_token'] = new_token['refresh_token']
            request.session['expires_at'] = new_token['expires_at']
            
            if athlete_details:
                request.session['athlete_first_name'] = athlete_details.get('firstname')
                request.session['athlete_last_name'] = athlete_details.get('lastname')
                request.session['athlete_profile'] = athlete_details.get('profile')
                request.session['measurement_preference'] = athlete_details.get('measurement_preference')
                request.session['last_profile_check'] = current_time
            
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Token updated successfully")
            return True
        
        elif is_profile_stale:
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Profile stale, fetching fresh details")
            athlete_details = await fetch_athlete_details(request.session['access_token'])
            if athlete_details:
                request.session['athlete_first_name'] = athlete_details.get('firstname')
                request.session['athlete_last_name'] = athlete_details.get('lastname')
                request.session['athlete_profile'] = athlete_details.get('profile')
                request.session['measurement_preference'] = athlete_details.get('measurement_preference')
                request.session['last_profile_check'] = current_time
                
                try:
                    if athlete_id != 'Unknown':
                        result = await db.execute(select(Athletes).where(Athletes.athlete_id == athlete_id))
                        athlete = result.scalar_one_or_none()
                        if athlete:
                            athlete.update_from_athlete_details(athlete_details)
                            await db.commit()
                except Exception as e:
                    logger.error(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] DB update failed for profile: {e}")
            
            return True

    return True


async def get_authenticated_athlete(request: Request, db: AsyncSession) -> Optional[Athletes]:
    """Get authenticated athlete. Updates session directly. Returns athlete or None."""
    if 'athlete_id' not in request.session:
        return None
    
    result = await db.execute(select(Athletes).where(Athletes.athlete_id == request.session['athlete_id']))
    athlete = result.scalar_one_or_none()
    
    if not athlete:
        request.session.clear()
        return None
    
    if not await ensure_valid_token(request, db):
        return None
    
    return athlete


# --- Routes ---

@app.get("/login")
@limiter.limit("5/minute")
async def login(request: Request):
    try:
        request.session.pop('logged_out', None)
        
        callback_url = str(request.url_for('callback'))
        logger.info(f"Generated callback URL: {callback_url}")
        
        state = secrets.token_urlsafe(16)
        request.session['oauth_state'] = state
        
        params = {
            "client_id": CLIENT_ID,
            "redirect_uri": callback_url,
            "response_type": "code",
            "scope": "read,activity:read_all,profile:read_all",
            "state": state
        }
        authorization_url = f"{AUTH_BASE_URL}?{urllib.parse.urlencode(params)}"
        
        return RedirectResponse(authorization_url, status_code=303)
    except Exception as e:
        logger.error(f"Login error: {e}")
        return JSONResponse({"error": "Authentication failed"}, status_code=401)


@app.get("/callback")
@limiter.limit("5/minute")
async def callback(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        if 'error' in request.query_params:
            logger.info(f"User denied Strava authorization - IP: {get_real_ip(request)}")
            return RedirectResponse('/', status_code=303)
        
        state = request.query_params.get('state')
        if not state or state != request.session.get('oauth_state'):
            logger.error(f"Invalid state parameter - IP: {get_real_ip(request)}")
            return RedirectResponse('/', status_code=303)
        
        code = request.query_params.get('code')
        if not code:
            logger.error(f"Missing code parameter - IP: {get_real_ip(request)}")
            return RedirectResponse('/', status_code=303)
        
        callback_url = str(request.url_for('callback'))
        logger.info(f"Using dynamic callback URL in callback handler: {callback_url}")
        
        try:
            token_response = await http_client.post(
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
            return RedirectResponse('/', status_code=303)
        
        athlete_data = token.get('athlete', {})
        athlete_id = athlete_data.get('id')
        logger.info(f"Athlete Data: {athlete_data}")
        
        if not athlete_id:
            logger.error('No athlete ID in token response')
            return RedirectResponse('/', status_code=303)
        
        try:
            result = await db.execute(select(Athletes).where(Athletes.athlete_id == athlete_id))
            athlete = result.scalar_one_or_none()
            
            if not athlete:
                athlete = Athletes(athlete_id=athlete_id)
                db.add(athlete)
                log_prefix = "New auth"
            else:
                log_prefix = "Existing auth"
            
            athlete.update_from_token(token, athlete_data)
            athlete_details = await fetch_athlete_details(token['access_token'])
            if athlete_details:
                athlete.update_from_athlete_details(athlete_details)
            
            await db.commit()
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Login: {log_prefix} - Name: {athlete_data.get('firstname')} {athlete_data.get('lastname')}")

            # Update session directly
            request.session['athlete_id'] = athlete_id
            request.session['athlete_username'] = athlete_data.get('username')
            request.session['athlete_first_name'] = athlete_data.get('firstname')
            request.session['athlete_last_name'] = athlete_data.get('lastname')
            request.session['athlete_profile'] = athlete_data.get('profile_medium')
            request.session['access_token'] = token['access_token']
            request.session['refresh_token'] = token['refresh_token']
            request.session['expires_at'] = token['expires_at']
            request.session['date_preference'] = athlete_data.get('date_preference')
            request.session['measurement_preference'] = athlete_data.get('measurement_preference')
            generate_csrf_token(request)
            
        except Exception as db_error:
            await db.rollback()
            logger.error(f'Database error during callback: {str(db_error)}')
            return RedirectResponse('/', status_code=303)
        
        return RedirectResponse('/', status_code=303)
        
    except Exception as e:
        logger.error(f'[{get_real_ip(request)}] OAuth callback error: {str(e)}')
        return RedirectResponse('/', status_code=303)


@app.post("/logout")
async def logout(request: Request):
    logger.info(f"[{get_real_ip(request)}] [Athlete: {request.session.get('athlete_id')}] Athlete logged out")
    request.session.clear()
    request.session['logged_out'] = True
    
    return RedirectResponse('/', status_code=303)


@app.get("/")
async def index(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        csrf_token = generate_csrf_token(request)
        user_count = await get_rounded_user_count(db)
        
        athlete = await get_authenticated_athlete(request, db)
        
        if not athlete:
            logger.info(f"[{get_real_ip(request)}] Index: Guest")
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "authenticated": False,
                    "csrf_token": csrf_token,
                    "user_count": user_count
                }
            )
        
        logger.info(f"[{get_real_ip(request)}] [Athlete: {request.session.get('athlete_id')}] Index: Auth")
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "authenticated": True,
                "athlete_id": request.session.get("athlete_id"),
                "athlete_first_name": request.session.get("athlete_first_name"),
                "athlete_last_name": request.session.get("athlete_last_name"),
                "athlete_profile": request.session.get("athlete_profile"),
                "csrf_token": csrf_token,
                "user_count": user_count
            }
        )
        
    except Exception as e:
        logger.error(f"[{get_real_ip(request)}] Index error: {str(e)}")
        request.session.clear()
        csrf_token = generate_csrf_token(request)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "authenticated": False,
                "csrf_token": csrf_token,
                "user_count": 200
            }
        )


@app.get("/customize")
@limiter.limit("100/hour")
async def customize(request: Request, db: AsyncSession = Depends(get_db)):
    athlete = await get_authenticated_athlete(request, db)
    
    if not athlete:
        if request.session.get('logged_out'):
            return templates.TemplateResponse("auth_required.html", {"request": request})
        return RedirectResponse('/login', status_code=303)
    
    logger.info(f"[{get_real_ip(request)}] [Athlete: {request.session['athlete_id']}] Customize page accessed")
    
    return templates.TemplateResponse(
        "customize.html",
        {
            "request": request,
            "authenticated": True,
            "athlete_id": request.session.get("athlete_id"),
            "athlete_first_name": request.session.get("athlete_first_name"),
            "athlete_last_name": request.session.get("athlete_last_name"),
            "athlete_profile": request.session.get("athlete_profile"),
            "measurement_preference": request.session.get("measurement_preference"),
            "csrf_token": request.session.get('csrf_token')
        }
    )


@app.get("/activities")
@limiter.limit("100/hour")
async def activities_page(request: Request, db: AsyncSession = Depends(get_db)):
    athlete = await get_authenticated_athlete(request, db)
    
    if not athlete:
        if request.session.get('logged_out'):
            return templates.TemplateResponse("auth_required.html", {"request": request})
        return RedirectResponse('/login', status_code=303)
    
    return templates.TemplateResponse(
        "activities.html",
        {
            "request": request,
            "authenticated": True,
            "athlete_id": request.session.get("athlete_id"),
            "athlete_first_name": request.session.get("athlete_first_name"),
            "athlete_last_name": request.session.get("athlete_last_name"),
            "athlete_profile": request.session.get("athlete_profile"),
            "measurement_preference": request.session.get("measurement_preference"),
            "csrf_token": request.session.get('csrf_token')
        }
    )


def create_sync_response(activities_data, page, per_page, last_synced, seconds_remaining, warning=None, using_cached=False):
    response = {
        "activities": activities_data,
        "pagination": {"page": page, "per_page": per_page},
        "cooldown": {
            "active": seconds_remaining > 0,
            "seconds_remaining": seconds_remaining,
            "total_cooldown": ActivityLists.SYNC_COOLDOWN.total_seconds()
        },
        "cached": using_cached
    }
    if last_synced:
        response["last_synced"] = last_synced.isoformat()
    else:
        response["last_synced"] = datetime.now(timezone.utc).isoformat()
    if warning:
        response["warning"] = warning
    return response


@app.get("/api/activities/sync")
@limiter.limit("100/hour")
async def sync_activities(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(30, le=30),
    db: AsyncSession = Depends(get_db)
):
    athlete = await get_authenticated_athlete(request, db)
    
    if not athlete:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    
    # Protect against extensive scraping/drive-by syncs
    if not validate_csrf_token(request):
        logger.warning(f"CSRF validation failed for sync - IP: {get_real_ip(request)}")
        return JSONResponse({"error": "Invalid CSRF token"}, status_code=403)
    
    athlete_id = request.session['athlete_id']
    
    per_page = min(per_page, ActivityLists.ITEMS_PER_PAGE)
    
    sync_instance = ActivityLists(athlete_id=athlete_id, page=page, per_page=per_page, data={})
    
    result = await db.execute(
        select(ActivityLists).where(
            ActivityLists.athlete_id == athlete_id,
            ActivityLists.page == page,
            ActivityLists.per_page == per_page
        )
    )
    sync_log = result.scalar_one_or_none()
    
    seconds_remaining = await sync_instance.get_cooldown_remaining(db)
    
    # Generate ETag
    etag = None
    if sync_log and sync_log.last_synced:
        etag_source = f"{athlete_id}:{page}:{per_page}:{sync_log.last_synced.isoformat()}"
        etag = hashlib.md5(etag_source.encode()).hexdigest()
    
    # Check cooldown
    sync_allowed = await sync_instance.is_sync_allowed(db)
    
    if not sync_allowed and sync_log:
        if etag and request.headers.get('If-None-Match') == etag:
            response = Response(content='', status_code=304)
            response.headers['ETag'] = etag
            response.headers['Cache-Control'] = 'private, max-age=180'
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Sync check: Not modified (304)")
            return response
        
        response_data = create_sync_response(
            sync_log.data if sync_log else [], page, per_page,
            sync_log.last_synced if sync_log else None, seconds_remaining, using_cached=True
        )
        response = JSONResponse(response_data)
        if etag:
            response.headers['ETag'] = etag
        response.headers['Cache-Control'] = 'private, max-age=180'
        return response
    
    try:
        api_response = await http_client.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers={"Authorization": f"Bearer {request.session['access_token']}"},
            params={"page": page, "per_page": per_page},
            timeout=15.0
        )
        
        if not api_response.is_success:
            response_data = create_sync_response(
                sync_log.data if sync_log else [], page, per_page,
                sync_log.last_synced if sync_log else None, seconds_remaining,
                warning="Failed to fetch fresh data, showing cached data" if sync_log else None,
                using_cached=True
            )
            status_code = api_response.status_code if not sync_log else 200
            response = JSONResponse(response_data, status_code=status_code)
            if etag:
                response.headers['ETag'] = etag
            response.headers['Cache-Control'] = 'private, max-age=180'
            return response
        
        activities_data = api_response.json()
        current_time = datetime.now(timezone.utc)
        
        try:
            if sync_log:
                sync_log.data = activities_data
                sync_log.last_synced = current_time
            else:
                sync_log = ActivityLists(
                    athlete_id=athlete_id,
                    data=activities_data,
                    page=page,
                    per_page=per_page,
                    last_synced=current_time
                )
                db.add(sync_log)
            
            await db.commit()
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Sync: Success (page {page})")

            
            response_data = create_sync_response(activities_data, page, per_page, current_time, 0, using_cached=False)
            response = JSONResponse(response_data)
            
            etag_source = f"{athlete_id}:{page}:{per_page}:{current_time.isoformat()}"
            etag = hashlib.md5(etag_source.encode()).hexdigest()
            response.headers['ETag'] = etag
            response.headers['Cache-Control'] = 'private, max-age=180'
            return response
            
        except Exception as db_error:
            await db.rollback()
            logger.error(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Sync database error: {db_error}")
            response_data = create_sync_response(activities_data, page, per_page, None, 0, using_cached=False)
            response = JSONResponse(response_data)
            response.headers['Cache-Control'] = 'private, max-age=180'
            return response
            
    except Exception as e:
        logger.error(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Sync unexpected error: {e}")
        response_data = create_sync_response(
            sync_log.data if sync_log else [], page, per_page,
            sync_log.last_synced if sync_log else None, seconds_remaining,
            warning="Failed to sync activities", using_cached=True
        )
        response = JSONResponse(response_data, status_code=500)
        if sync_log and sync_log.last_synced:
            etag_source = f"{athlete_id}:{page}:{per_page}:{sync_log.last_synced.isoformat()}"
            etag = hashlib.md5(etag_source.encode()).hexdigest()
            response.headers['ETag'] = etag
        response.headers['Cache-Control'] = 'private, max-age=180'
        return response


@app.get("/api/activities/{activity_id}")
@limiter.limit("100/hour")
async def get_activity(
    request: Request,
    activity_id: int,
    db: AsyncSession = Depends(get_db)
):
    athlete = await get_authenticated_athlete(request, db)
    
    if not athlete:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    
    athlete_id = request.session['athlete_id']
    
    result = await db.execute(
        select(Activities).where(
            Activities.athlete_id == athlete_id,
            Activities.activity_id == activity_id
        )
    )
    activity = result.scalar_one_or_none()
    
    # Generate ETag
    etag = None
    if activity and activity.last_synced:
        etag_source = f"{athlete_id}:{activity_id}:{activity.last_synced.isoformat()}"
        etag = hashlib.md5(etag_source.encode()).hexdigest()
    
    if activity:
        seconds_remaining = await activity.get_cooldown_remaining(db)
        if seconds_remaining > 0:
            if etag and request.headers.get('If-None-Match') == etag:
                response = Response(content='', status_code=304)
                response.headers['ETag'] = etag
                response.headers['Cache-Control'] = 'private, max-age=300'
                logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Activity {activity_id} check: Not modified (304)")
                return response
            
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Activity {activity_id}: Cache hit ({seconds_remaining}s cooldown)")
            response_data = {
                "activity": activity.data,
                "cooldown": {
                    "active": True,
                    "seconds_remaining": seconds_remaining,
                    "total_cooldown": Activities.SYNC_COOLDOWN.total_seconds()
                },
                "cached": True
            }
            response = JSONResponse(response_data)
            if etag:
                response.headers['ETag'] = etag
            response.headers['Cache-Control'] = 'private, max-age=300'
            return response
    
    try:
        api_response = await http_client.get(
            f"https://www.strava.com/api/v3/activities/{activity_id}",
            headers={"Authorization": f"Bearer {request.session['access_token']}"},
            timeout=15.0
        )
        
        if not api_response.is_success:
            logger.error(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Strava API error for activity {activity_id}: {api_response.status_code}")
            if activity:
                seconds_remaining = await activity.get_cooldown_remaining(db)
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
                response = JSONResponse(response_data, status_code=api_response.status_code)
                if etag:
                    response.headers['ETag'] = etag
                response.headers['Cache-Control'] = 'private, max-age=300'
                return response
            return JSONResponse({"error": "Failed to fetch activity data"}, status_code=api_response.status_code)
        
        activity_data = api_response.json()
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
                db.add(activity)
            
            await db.commit()
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Activity {activity_id}: Fresh fetch & cached")
            
            response_data = {
                "activity": activity_data,
                "cooldown": {
                    "active": False,
                    "seconds_remaining": 0,
                    "total_cooldown": Activities.SYNC_COOLDOWN.total_seconds()
                },
                "cached": False
            }
            response = JSONResponse(response_data)
            
            etag_source = f"{athlete_id}:{activity_id}:{current_time.isoformat()}"
            etag = hashlib.md5(etag_source.encode()).hexdigest()
            response.headers['ETag'] = etag
            response.headers['Cache-Control'] = 'private, max-age=300'
            return response
            
        except Exception as db_error:
            await db.rollback()
            logger.error(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Database error caching activity {activity_id}: {str(db_error)}")
            raise
            
    except Exception as e:
        logger.error(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Error fetching activity {activity_id}: {str(e)}")
        if activity:
            seconds_remaining = await activity.get_cooldown_remaining(db)
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
            response = JSONResponse(response_data, status_code=500)
            if activity.last_synced:
                etag_source = f"{athlete_id}:{activity_id}:{activity.last_synced.isoformat()}"
                etag = hashlib.md5(etag_source.encode()).hexdigest()
                response.headers['ETag'] = etag
            response.headers['Cache-Control'] = 'private, max-age=300'
            return response
        return JSONResponse({"error": "Failed to fetch activity data"}, status_code=500)


async def delete_user_data(db: AsyncSession, athlete_id: int):
    try:
        logger.info(f"Deleting all data for athlete {athlete_id}")
        await db.execute(delete(Activities).where(Activities.athlete_id == athlete_id))
        await db.execute(delete(ActivityLists).where(ActivityLists.athlete_id == athlete_id))
        await db.execute(delete(Athletes).where(Athletes.athlete_id == athlete_id))
        await db.commit()
        logger.info(f"Successfully deleted all data for athlete {athlete_id}")
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting data for athlete {athlete_id}: {str(e)}")
        raise


@app.get("/webhook")
@limiter.limit("10/minute")
async def webhook_verify(
    request: Request,
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if not VERIFY_TOKEN:
        logger.error(f"[{get_real_ip(request)}] Webhook configuration error: STRAVA_VERIFY_TOKEN missing")
        return JSONResponse({"error": "Webhook not configured"}, status_code=500)
    
    if hub_verify_token == VERIFY_TOKEN:
        return JSONResponse({"hub.challenge": hub_challenge})
    
    logger.warning(f"[{get_real_ip(request)}] Webhook verification failed: Invalid token")
    return JSONResponse({"error": "Invalid verification token"}, status_code=403)


@app.post("/webhook")
@limiter.limit("10/minute")
async def webhook_event(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        signature = request.headers.get('X-Strava-Signature')
        if not signature:
            if ENVIRONMENT == "dev":
                logger.warning(f"[{get_real_ip(request)}] Webhook: Skipping signature verification (dev mode)")
            else:
                logger.warning(f"[{get_real_ip(request)}] Webhook: Missing Strava signature")
                return JSONResponse({"error": "Unauthorized"}, status_code=403)
        
        event = await request.json()
        logger.info(f"[{get_real_ip(request)}] Webhook event received: {event.get('object_type')} {event.get('aspect_type')}")
        
        if (event.get("object_type") == "athlete" and
            event.get("aspect_type") == "update" and
            event.get("updates", {}).get("authorized") == "false"):
            athlete_id = event.get("owner_id")
            if athlete_id:
                await delete_user_data(db, athlete_id)
                logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Webhook: Deauthorization processed")
            else:
                logger.warning(f"[{get_real_ip(request)}] Webhook: Deauthorization missing athlete_id")
        
        return JSONResponse({"status": "ok"})
    except Exception as e:
        logger.error(f"[{get_real_ip(request)}] Webhook processing error: {str(e)}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.exception_handler(404)
async def page_not_found(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)


@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse('/static/images/FitnessOverlaysLogo.ico', status_code=302)


@app.get("/robots.txt")
async def robots_txt():
    return RedirectResponse('/static/robots.txt', status_code=302)


@app.get("/llms.txt")
async def llms_txt():
    return RedirectResponse('/static/llms.txt', status_code=302)


@app.get("/demo")
@limiter.limit("30/hour")
async def demo(request: Request, db: AsyncSession = Depends(get_db)):
    logger.info(f"[{get_real_ip(request)}] Demo page accessed")
    
    activity = DEMO_ACTIVITY_DATA
    is_authenticated = False
    athlete_id = request.session.get("athlete_id")
    
    if athlete_id:
        result = await db.execute(select(Athletes).where(Athletes.athlete_id == athlete_id))
        athlete = result.scalar_one_or_none()
        if athlete:
            is_authenticated = True
            logger.info(f"[{get_real_ip(request)}] [Athlete: {athlete_id}] Demo accessed by authenticated user")
    
    csrf_token = request.session.get('csrf_token') if is_authenticated else generate_csrf_token(request)
    
    return templates.TemplateResponse(
        'customize.html',
        {
            "request": request,
            "activity": activity,
            "demo_mode": True,
            "authenticated": is_authenticated,
            "athlete_id": request.session.get("athlete_id") if is_authenticated else None,
            "athlete_first_name": request.session.get("athlete_first_name") if is_authenticated else None,
            "athlete_last_name": request.session.get("athlete_last_name") if is_authenticated else None,
            "athlete_profile": request.session.get("athlete_profile") if is_authenticated else None,
            "csrf_token": csrf_token
        }
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


@app.get("/sitemap.xml")
async def sitemap_xml():
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
    return Response(content=sitemap, media_type='application/xml')


if __name__ == '__main__':
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=DEBUG_MODE)
