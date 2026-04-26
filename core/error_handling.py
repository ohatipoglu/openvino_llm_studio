"""
core/error_handling.py

Unified error handling strategy for the application.

Provides:
- Custom exception hierarchy
- Error context capture
- Automatic logging
- User-friendly error messages
- Retry decorators
"""

import logging
import functools
import traceback
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# EXCEPTION HIERARCHY
# ═══════════════════════════════════════════════════════════════════

class AppError(Exception):
    """Base application exception."""
    
    def __init__(self, message: str, context: dict = None,
                 original_exception: Exception = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_exception = original_exception
        self.stack_trace = traceback.format_exc()


class ConfigurationError(AppError):
    """Configuration or setup error."""
    pass


class ModelError(AppError):
    """Model loading or inference error."""
    pass


class SearchError(AppError):
    """Web search error."""
    pass


class DSPyError(AppError):
    """DSPy optimization error."""
    pass


class SecurityError(AppError):
    """Security validation error."""
    pass


class BackendError(AppError):
    """Backend communication error."""
    pass


class ResourceError(AppError):
    """Resource exhaustion error (memory, disk, etc.)."""
    pass


class ValidationError(AppError):
    """Input validation error."""
    pass


# ═══════════════════════════════════════════════════════════════════
# ERROR CONTEXT
# ═══════════════════════════════════════════════════════════════════

class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """
    Comprehensive error context for debugging.
    
    Captures:
    - Module and function where error occurred
    - Input parameters
    - System state
    - Stack trace
    """
    module: str
    function: str
    error_type: str
    error_message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    parameters: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    stack_trace: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        import datetime
        if not self.timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat()
        if not self.stack_trace:
            self.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "module": self.module,
            "function": self.function,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "parameters": self.parameters,
            "state": self.state,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLER
# ═══════════════════════════════════════════════════════════════════

class ErrorHandler:
    """
    Centralized error handler.
    
    Features:
    - Automatic context capture
    - Logging integration
    - User-friendly messages
    - Error recovery strategies
    """
    
    # User-friendly error messages
    USER_MESSAGES = {
        ModelError: "Model işlemi sırasında bir hata oluştu. Lütfen modeli yeniden yükleyin.",
        SearchError: "Arama motoru hatası. Lütfen internet bağlantınızı kontrol edin.",
        DSPyError: "Prompt işleme hatası. Varsayılan mod kullanılıyor.",
        SecurityError: "Güvenlik kontrolü başarısız. İstek işlenemedi.",
        BackendError: "Backend bağlantı hatası. Lütfen backend durumunu kontrol edin.",
        ResourceError: "Yetersiz sistem kaynağı. Lütfen diğer uygulamaları kapatın.",
        ValidationError: "Geçersiz girdi. Lütfen parametreleri kontrol edin.",
    }
    
    # Recovery strategies
    RECOVERY_STRATEGIES = {
        ModelError: "reload_model",
        SearchError: "fallback_to_local",
        DSPyError: "use_default_template",
        BackendError: "switch_backend",
        ResourceError: "unload_idle_models",
    }
    
    @classmethod
    def handle(cls, error: Exception, context: ErrorContext) -> dict:
        """
        Handle error with context.
        
        Args:
            error: Caught exception
            context: Error context
        
        Returns:
            Error response dict for API/UI
        """
        # Log error
        cls._log_error(error, context)
        
        # Get user message
        user_message = cls._get_user_message(error)
        
        # Determine recovery action
        recovery = cls._get_recovery_strategy(error)
        
        return {
            "success": False,
            "error_type": type(error).__name__,
            "user_message": user_message,
            "technical_message": str(error),
            "recovery_action": recovery,
            "context": context.to_dict(),
        }
    
    @classmethod
    def _log_error(cls, error: Exception, context: ErrorContext):
        """Log error with appropriate level."""
        log_method = getattr(logger, context.severity.value)
        
        log_method(
            f"{context.module}.{context.function}: "
            f"{type(error).__name__}: {error}",
            exc_info=True,
            extra={"error_context": context.to_dict()}
        )
    
    @classmethod
    def _get_user_message(cls, error: Exception) -> str:
        """Get user-friendly error message."""
        for error_type, message in cls.USER_MESSAGES.items():
            if isinstance(error, error_type):
                return message
        return f"Beklenmeyen hata: {type(error).__name__}"
    
    @classmethod
    def _get_recovery_strategy(cls, error: Exception) -> Optional[str]:
        """Get recovery strategy for error type."""
        for error_type, strategy in cls.RECOVERY_STRATEGIES.items():
            if isinstance(error, error_type):
                return strategy
        return None


# ═══════════════════════════════════════════════════════════════════
# RETRY DECORATOR
# ═══════════════════════════════════════════════════════════════════

def retry_on_error(
    exceptions: tuple = (Exception,),
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger_name: str = __name__
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        exceptions: Exception types to catch
        max_retries: Maximum retry attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        logger_name: Logger name
    
    Returns:
        Decorated function
    
    Usage:
        @retry_on_error(exceptions=(ConnectionError,), max_retries=3)
        def api_call():
            ...
    """
    import time
    import functools
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retry_delay = delay
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    
                    if attempt == max_retries:
                        break
                    
                    log = logging.getLogger(logger_name)
                    log.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                        f"failed: {e}. Retrying in {retry_delay:.1f}s..."
                    )
                    
                    time.sleep(retry_delay)
                    retry_delay *= backoff
            
            raise last_error
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════
# CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════

class error_context:
    """
    Context manager for error handling.
    
    Automatically captures context and handles errors.
    
    Usage:
        with error_context("MyModule", "my_function", params={"x": 1}):
            # Code that might raise
            ...
    """
    
    def __init__(self, module: str, function: str,
                 params: dict = None, state: dict = None):
        self.module = module
        self.function = function
        self.params = params or {}
        self.state = state or {}
        self.context = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.context = ErrorContext(
                module=self.module,
                function=self.function,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                parameters=self.params,
                state=self.state,
            )
            
            # Handle error
            result = ErrorHandler.handle(exc_val, self.context)
            
            # Log to database if available
            self._log_to_database(result)
            
            # Don't suppress exception - just enhance it
            if hasattr(exc_val, 'error_context'):
                exc_val.error_context = result
            else:
                exc_val.error_context = result
        
        return False  # Don't suppress exceptions
    
    def _log_to_database(self, error_result: dict):
        """Log error to database if available."""
        try:
            from modules.database import DatabaseManager
            db = DatabaseManager()
            db.log_error(
                session_id=self.state.get("session_id", "unknown"),
                module=f"{self.module}.{self.function}",
                error=Exception(error_result["technical_message"]),
                context=error_result["context"]
            )
        except Exception as e:
            logger.warning(f"Failed to log error to database: {e}")
