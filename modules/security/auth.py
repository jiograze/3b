"""Authentication and security module for Ötüken3D."""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import time
from pydantic import BaseModel

from ..core.logger import setup_logger
from ..core.exceptions import SecurityError

logger = setup_logger(__name__)

# Security configurations
SECRET_KEY = "your-secret-key-here"  # Should be loaded from environment variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Rate limiting configurations
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS = 100  # requests per window

# API Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: list[str] = []

class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    scopes: list[str] = []

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed based on rate limits."""
        current_time = time.time()
        client_requests = self.requests.get(client_id, [])
        
        # Remove old requests
        client_requests = [req_time for req_time in client_requests
                         if current_time - req_time < RATE_LIMIT_WINDOW]
        
        # Check rate limit
        if len(client_requests) >= MAX_REQUESTS:
            return False
        
        # Add new request
        client_requests.append(current_time)
        self.requests[client_id] = client_requests
        return True

class SecurityManager:
    """Security management utility."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.api_keys = set()  # Should be loaded from secure storage
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        try:
            to_encode = data.copy()
            
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=15)
            
            to_encode.update({"exp": expire})
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            return encoded_jwt
            
        except Exception as e:
            raise SecurityError(f"Failed to create access token: {str(e)}")
    
    def verify_token(self, token: str) -> TokenData:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials"
                )
            token_scopes = payload.get("scopes", [])
            return TokenData(username=username, scopes=token_scopes)
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    def generate_api_key(self) -> str:
        """Generate new API key."""
        api_key = secrets.token_urlsafe(32)
        self.api_keys.add(api_key)
        return api_key
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key."""
        return api_key in self.api_keys
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limits."""
        return self.rate_limiter.is_allowed(client_id)

class InputValidator:
    """Input validation utility."""
    
    @staticmethod
    def validate_text_input(text: str) -> bool:
        """Validate text input for text-to-3D."""
        if not text or len(text) > 1000:
            return False
        # Add more validation rules as needed
        return True
    
    @staticmethod
    def validate_image_input(image_data: bytes) -> bool:
        """Validate image input for image-to-3D."""
        # Implement image validation
        # Check file size, format, dimensions, etc.
        return True
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input."""
        # Implement text sanitization
        return text.strip()

# Dependency functions for FastAPI
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    security_manager: SecurityManager = Depends()
) -> User:
    """Get current user from token."""
    token_data = security_manager.verify_token(token)
    # Here you would typically load user data from database
    user = User(
        username=token_data.username,
        scopes=token_data.scopes
    )
    return user

async def verify_api_key(
    api_key: str = Depends(API_KEY_HEADER),
    security_manager: SecurityManager = Depends()
) -> bool:
    """Verify API key."""
    if not security_manager.verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return True

async def check_rate_limit(
    client_id: str,
    security_manager: SecurityManager = Depends()
) -> bool:
    """Check rate limit."""
    if not security_manager.check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return True

# Security middleware
class SecurityMiddleware:
    """Security middleware for FastAPI."""
    
    def __init__(self):
        self.security_manager = SecurityManager()
        self.input_validator = InputValidator()
    
    async def authenticate(
        self,
        token: str = Depends(oauth2_scheme)
    ) -> User:
        """Authenticate request."""
        return await get_current_user(token, self.security_manager)
    
    async def validate_api_key(
        self,
        api_key: str = Depends(API_KEY_HEADER)
    ) -> bool:
        """Validate API key."""
        return await verify_api_key(api_key, self.security_manager)
    
    async def check_rate_limit(
        self,
        client_id: str
    ) -> bool:
        """Check rate limit."""
        return await check_rate_limit(client_id, self.security_manager)
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data."""
        if isinstance(data, str):
            return self.input_validator.validate_text_input(data)
        elif isinstance(data, bytes):
            return self.input_validator.validate_image_input(data)
        return False 