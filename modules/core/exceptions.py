"""Custom exceptions for Ötüken3D."""

class Otuken3DError(Exception):
    """Base exception class for Ötüken3D."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ModelError(Otuken3DError):
    """Exception raised for errors in the model operations."""
    pass

class ProcessingError(Otuken3DError):
    """Exception raised for errors in data processing."""
    pass

class ConfigError(Otuken3DError):
    """Exception raised for configuration errors."""
    pass

class ValidationError(Otuken3DError):
    """Exception raised for data validation errors."""
    pass

class ResourceError(Otuken3DError):
    """Exception raised for resource-related errors (GPU, memory, etc.)."""
    pass

class APIError(Otuken3DError):
    """Exception raised for API-related errors."""
    pass

class SecurityError(Otuken3DError):
    """Exception raised for security-related issues."""
    pass

class DataError(Otuken3DError):
    """Exception raised for data-related issues."""
    pass

class NetworkError(Otuken3DError):
    """Exception raised for network-related issues."""
    pass 