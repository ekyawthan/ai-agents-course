# Security & Compliance

## Introduction to Agent Security

Security is critical for production agent systems. This section covers authentication, authorization, data protection, and compliance requirements.

### Security Principles

**Defense in Depth**: Multiple layers of security
**Least Privilege**: Minimum necessary access
**Zero Trust**: Verify everything
**Encryption**: Protect data at rest and in transit
**Audit Everything**: Complete logging

### Threat Model

**Threats**:
- Unauthorized access
- Data breaches
- Prompt injection
- Model manipulation
- Resource exhaustion
- Privacy violations

## Authentication and Authorization

### JWT-Based Authentication

```python
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict

class AuthManager:
    """JWT-based authentication"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer()
    
    def create_token(self, 
                    user_id: str,
                    roles: List[str],
                    expires_in: int = 3600) -> str:
        """Create JWT token"""
        
        payload = {
            "user_id": user_id,
            "roles": roles,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def get_current_user(self,
                              credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
        """Get current user from token"""
        
        token = credentials.credentials
        payload = self.verify_token(token)
        
        return {
            "user_id": payload["user_id"],
            "roles": payload["roles"]
        }

# Role-Based Access Control
class RBACManager:
    """Role-based access control"""
    
    def __init__(self):
        self.permissions = {
            "admin": ["read", "write", "delete", "admin"],
            "user": ["read", "write"],
            "viewer": ["read"]
        }
    
    def has_permission(self, roles: List[str], required_permission: str) -> bool:
        """Check if roles have required permission"""
        
        for role in roles:
            if role in self.permissions:
                if required_permission in self.permissions[role]:
                    return True
        
        return False
    
    def require_permission(self, permission: str):
        """Decorator to require permission"""
        
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Get user from context
                user = kwargs.get('current_user')
                
                if not user:
                    raise HTTPException(status_code=401, detail="Not authenticated")
                
                if not self.has_permission(user['roles'], permission):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator

# Secure Agent API
class SecureAgentAPI:
    """Agent API with authentication"""
    
    def __init__(self):
        self.app = FastAPI()
        self.auth = AuthManager(secret_key="your-secret-key")
        self.rbac = RBACManager()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup secure routes"""
        
        @self.app.post("/auth/login")
        async def login(credentials: LoginRequest):
            """Login and get token"""
            # Verify credentials (simplified)
            if self.verify_credentials(credentials.username, credentials.password):
                token = self.auth.create_token(
                    user_id=credentials.username,
                    roles=["user"]
                )
                return {"token": token}
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        
        @self.app.post("/agent/process")
        async def process(
            request: AgentRequest,
            current_user: Dict = Depends(self.auth.get_current_user)
        ):
            """Process request (requires authentication)"""
            
            # Check permission
            if not self.rbac.has_permission(current_user['roles'], 'write'):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            # Process request
            result = await self.process_request(request, current_user)
            return {"result": result}
    
    def verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials"""
        # In production, check against database with hashed passwords
        return True

class LoginRequest(BaseModel):
    username: str
    password: str

# Usage
api = SecureAgentAPI()
```

### API Key Management

```python
import secrets
import hashlib
from datetime import datetime

class APIKeyManager:
    """Manage API keys"""
    
    def __init__(self):
        self.keys = {}  # In production, use database
    
    def generate_key(self, user_id: str, name: str) -> str:
        """Generate new API key"""
        
        # Generate secure random key
        key = f"sk_{secrets.token_urlsafe(32)}"
        
        # Hash for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Store
        self.keys[key_hash] = {
            "user_id": user_id,
            "name": name,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "usage_count": 0
        }
        
        return key
    
    def verify_key(self, key: str) -> Optional[Dict]:
        """Verify API key"""
        
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if key_hash in self.keys:
            # Update usage
            self.keys[key_hash]["last_used"] = datetime.utcnow()
            self.keys[key_hash]["usage_count"] += 1
            
            return self.keys[key_hash]
        
        return None
    
    def revoke_key(self, key: str):
        """Revoke API key"""
        
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if key_hash in self.keys:
            del self.keys[key_hash]
            return True
        
        return False

# API Key Authentication
from fastapi.security import APIKeyHeader

class APIKeyAuth:
    """API Key authentication"""
    
    def __init__(self, key_manager: APIKeyManager):
        self.key_manager = key_manager
        self.api_key_header = APIKeyHeader(name="X-API-Key")
    
    async def verify(self, api_key: str = Security(APIKeyHeader(name="X-API-Key"))):
        """Verify API key"""
        
        key_data = self.key_manager.verify_key(api_key)
        
        if not key_data:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return key_data

# Usage
key_manager = APIKeyManager()
api_key = key_manager.generate_key("user123", "Production Key")
print(f"API Key: {api_key}")
```

## Data Encryption

### Encryption at Rest

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64

class DataEncryption:
    """Encrypt sensitive data"""
    
    def __init__(self, password: str):
        self.key = self.derive_key(password)
        self.cipher = Fernet(self.key)
    
    def derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'static_salt',  # In production, use random salt
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt data"""
        
        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        
        encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()

# Encrypted Storage
class EncryptedStorage:
    """Store data with encryption"""
    
    def __init__(self, encryption_key: str):
        self.encryption = DataEncryption(encryption_key)
        self.storage = {}
    
    def store(self, key: str, value: str):
        """Store encrypted data"""
        
        encrypted_value = self.encryption.encrypt(value)
        self.storage[key] = encrypted_value
    
    def retrieve(self, key: str) -> Optional[str]:
        """Retrieve and decrypt data"""
        
        encrypted_value = self.storage.get(key)
        
        if encrypted_value:
            return self.encryption.decrypt(encrypted_value)
        
        return None

# Usage
storage = EncryptedStorage("my-secret-password")
storage.store("api_key", "sk_1234567890")
retrieved = storage.retrieve("api_key")
print(f"Retrieved: {retrieved}")
```

### Encryption in Transit (TLS/SSL)

```python
import ssl
from fastapi import FastAPI
import uvicorn

class SecureServer:
    """HTTPS server with TLS"""
    
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "Secure server"}
    
    def run(self, 
            host: str = "0.0.0.0",
            port: int = 443,
            cert_file: str = "cert.pem",
            key_file: str = "key.pem"):
        """Run with TLS"""
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            ssl_version=ssl.PROTOCOL_TLS,
            ssl_cert_reqs=ssl.CERT_REQUIRED
        )

# Generate self-signed certificate (for development only)
def generate_self_signed_cert():
    """Generate self-signed certificate"""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    
    # Generate certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Agent System"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).sign(private_key, hashes.SHA256())
    
    return private_key, cert
```

## Audit Logging

### Comprehensive Audit System

```python
import logging
from datetime import datetime
from typing import Optional
import json

class AuditLogger:
    """Audit logging system"""
    
    def __init__(self, log_file: str = "audit.log"):
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log_event(self,
                  event_type: str,
                  user_id: str,
                  action: str,
                  resource: str,
                  result: str,
                  metadata: Optional[Dict] = None):
        """Log audit event"""
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "result": result,
            "metadata": metadata or {},
            "ip_address": self.get_client_ip()
        }
        
        self.logger.info(json.dumps(event))
    
    def log_access(self, user_id: str, resource: str, granted: bool):
        """Log access attempt"""
        
        self.log_event(
            event_type="access",
            user_id=user_id,
            action="access",
            resource=resource,
            result="granted" if granted else "denied"
        )
    
    def log_data_access(self, user_id: str, data_type: str, operation: str):
        """Log data access"""
        
        self.log_event(
            event_type="data_access",
            user_id=user_id,
            action=operation,
            resource=data_type,
            result="success"
        )
    
    def log_security_event(self, user_id: str, event: str, severity: str):
        """Log security event"""
        
        self.log_event(
            event_type="security",
            user_id=user_id,
            action=event,
            resource="system",
            result=severity,
            metadata={"severity": severity}
        )
    
    def get_client_ip(self) -> str:
        """Get client IP address"""
        # In production, extract from request
        return "0.0.0.0"

# Audit Middleware
class AuditMiddleware:
    """Middleware for automatic audit logging"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
    
    async def __call__(self, request, call_next):
        """Process request with audit logging"""
        
        # Log request
        user_id = request.state.user_id if hasattr(request.state, 'user_id') else "anonymous"
        
        self.audit_logger.log_event(
            event_type="api_request",
            user_id=user_id,
            action=request.method,
            resource=request.url.path,
            result="started"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log success
            self.audit_logger.log_event(
                event_type="api_request",
                user_id=user_id,
                action=request.method,
                resource=request.url.path,
                result="success",
                metadata={"status_code": response.status_code}
            )
            
            return response
            
        except Exception as e:
            # Log failure
            self.audit_logger.log_event(
                event_type="api_request",
                user_id=user_id,
                action=request.method,
                resource=request.url.path,
                result="error",
                metadata={"error": str(e)}
            )
            
            raise

# Usage
audit_logger = AuditLogger()
audit_logger.log_access("user123", "/agent/process", granted=True)
audit_logger.log_security_event("user456", "failed_login", "warning")
```

## Regulatory Considerations

### GDPR Compliance

```python
class GDPRCompliance:
    """GDPR compliance features"""
    
    def __init__(self):
        self.data_store = {}
        self.consent_records = {}
        self.audit_logger = AuditLogger()
    
    def collect_consent(self, user_id: str, purposes: List[str]) -> bool:
        """Collect user consent"""
        
        self.consent_records[user_id] = {
            "purposes": purposes,
            "timestamp": datetime.utcnow(),
            "version": "1.0"
        }
        
        self.audit_logger.log_event(
            event_type="consent",
            user_id=user_id,
            action="collect",
            resource="consent",
            result="success",
            metadata={"purposes": purposes}
        )
        
        return True
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has consented"""
        
        consent = self.consent_records.get(user_id)
        
        if not consent:
            return False
        
        return purpose in consent["purposes"]
    
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data (right to data portability)"""
        
        self.audit_logger.log_event(
            event_type="data_export",
            user_id=user_id,
            action="export",
            resource="user_data",
            result="success"
        )
        
        # Collect all user data
        user_data = {
            "user_id": user_id,
            "data": self.data_store.get(user_id, {}),
            "consent": self.consent_records.get(user_id, {}),
            "exported_at": datetime.utcnow().isoformat()
        }
        
        return user_data
    
    def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data (right to be forgotten)"""
        
        self.audit_logger.log_event(
            event_type="data_deletion",
            user_id=user_id,
            action="delete",
            resource="user_data",
            result="success"
        )
        
        # Delete all user data
        if user_id in self.data_store:
            del self.data_store[user_id]
        
        if user_id in self.consent_records:
            del self.consent_records[user_id]
        
        return True
    
    def anonymize_data(self, user_id: str) -> bool:
        """Anonymize user data"""
        
        if user_id in self.data_store:
            # Replace with anonymized version
            self.data_store[f"anon_{hash(user_id)}"] = self.data_store[user_id]
            del self.data_store[user_id]
        
        return True

# Usage
gdpr = GDPRCompliance()

# Collect consent
gdpr.collect_consent("user123", ["analytics", "personalization"])

# Check consent
has_consent = gdpr.check_consent("user123", "analytics")

# Export data
user_data = gdpr.export_user_data("user123")

# Delete data
gdpr.delete_user_data("user123")
```

### SOC 2 Compliance

```python
class SOC2Compliance:
    """SOC 2 compliance controls"""
    
    def __init__(self):
        self.audit_logger = AuditLogger()
        self.access_controls = RBACManager()
    
    def implement_access_controls(self):
        """Implement access controls (Security)"""
        # Already implemented via RBAC
        pass
    
    def monitor_availability(self) -> Dict:
        """Monitor system availability (Availability)"""
        
        # Check service health
        health_status = {
            "agent_service": self.check_service_health("agent"),
            "tool_service": self.check_service_health("tools"),
            "memory_service": self.check_service_health("memory")
        }
        
        uptime = sum(1 for status in health_status.values() if status) / len(health_status)
        
        return {
            "uptime_percentage": uptime * 100,
            "services": health_status
        }
    
    def ensure_processing_integrity(self, data: Dict) -> bool:
        """Ensure processing integrity (Processing Integrity)"""
        
        # Validate data
        if not self.validate_data(data):
            return False
        
        # Log processing
        self.audit_logger.log_event(
            event_type="data_processing",
            user_id=data.get("user_id", "system"),
            action="process",
            resource="data",
            result="success"
        )
        
        return True
    
    def protect_confidentiality(self, data: str) -> str:
        """Protect data confidentiality (Confidentiality)"""
        
        encryption = DataEncryption("secret-key")
        return encryption.encrypt(data)
    
    def maintain_privacy(self, user_id: str) -> bool:
        """Maintain privacy (Privacy)"""
        
        # Implement privacy controls
        gdpr = GDPRCompliance()
        
        # Check consent
        has_consent = gdpr.check_consent(user_id, "data_processing")
        
        if not has_consent:
            return False
        
        return True
    
    def check_service_health(self, service: str) -> bool:
        """Check service health"""
        # In production, actually check service
        return True
    
    def validate_data(self, data: Dict) -> bool:
        """Validate data integrity"""
        # Implement validation logic
        return True
```

## Best Practices

1. **Authentication**: Always authenticate users
2. **Authorization**: Implement least privilege
3. **Encryption**: Encrypt sensitive data
4. **Audit logging**: Log all security events
5. **Input validation**: Validate all inputs
6. **Rate limiting**: Prevent abuse
7. **Security headers**: Use proper HTTP headers
8. **Regular updates**: Keep dependencies updated
9. **Security testing**: Regular penetration testing
10. **Incident response**: Have a plan

## Next Steps

You now understand security and compliance! Next, we'll explore cost optimization strategies for production agent systems.
