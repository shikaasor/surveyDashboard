"""
Logging configuration and utilities.

This module sets up structured logging for the application with proper
redaction of sensitive information like API tokens.
"""

import os
import logging
import logging.handlers
from typing import Any, Dict
import re
from datetime import datetime


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from logs."""
    
    SENSITIVE_PATTERNS = [
        # API tokens and keys
        (re.compile(r'Token\s+([a-zA-Z0-9]+)', re.IGNORECASE), 'Token [REDACTED]'),
        (re.compile(r'api_token[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9]+)', re.IGNORECASE), 'api_token=[REDACTED]'),
        (re.compile(r'password[\'"\s]*[:=][\'"\s]*([^\s\'"]+)', re.IGNORECASE), 'password=[REDACTED]'),
        
        # URLs with tokens
        (re.compile(r'(https?://[^/]+/[^?]*)\?[^&]*token=([^&\s]+)', re.IGNORECASE), r'\1?token=[REDACTED]'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to redact sensitive information."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            message = record.msg
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                message = pattern.sub(replacement, message)
            record.msg = message
        
        # Also filter args if present
        if hasattr(record, 'args') and record.args:
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.SENSITIVE_PATTERNS:
                        arg = pattern.sub(replacement, arg)
                filtered_args.append(arg)
            record.args = tuple(filtered_args)
        
        return True


def setup_logger(name: str = 'kobo_dashboard') -> logging.Logger:
    """
    Set up structured logging with sensitive data filtering.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set log level from environment
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SensitiveDataFilter())
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(SensitiveDataFilter())
    logger.addHandler(file_handler)
    
    return logger


def log_api_request(logger: logging.Logger, method: str, url: str, 
                   status_code: int = None, response_time: float = None) -> None:
    """
    Log API request details with sensitive data redaction.
    
    Args:
        logger: Logger instance
        method: HTTP method
        url: Request URL
        status_code: Response status code
        response_time: Response time in seconds
    """
    # Redact URL if it contains sensitive information
    filtered_url = url
    for pattern, replacement in SensitiveDataFilter.SENSITIVE_PATTERNS:
        filtered_url = pattern.sub(replacement, filtered_url)
    
    message = f"{method} {filtered_url}"
    if status_code:
        message += f" -> {status_code}"
    if response_time:
        message += f" ({response_time:.2f}s)"
    
    if status_code and status_code >= 400:
        logger.error(message)
    else:
        logger.info(message)


def log_data_processing(logger: logging.Logger, operation: str, 
                       record_count: int, processing_time: float = None) -> None:
    """
    Log data processing operations.
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        record_count: Number of records processed
        processing_time: Processing time in seconds
    """
    message = f"Data processing: {operation} - {record_count:,} records"
    if processing_time:
        message += f" in {processing_time:.2f}s"
    
    logger.info(message)


def log_error_with_context(logger: logging.Logger, error: Exception, 
                          context: Dict[str, Any] = None) -> None:
    """
    Log error with additional context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    message = f"Error: {type(error).__name__}: {str(error)}"
    
    if context:
        # Filter out sensitive information from context
        filtered_context = {}
        for key, value in context.items():
            if any(sensitive in key.lower() for sensitive in ['token', 'password', 'key']):
                filtered_context[key] = '[REDACTED]'
            else:
                filtered_context[key] = value
        
        if filtered_context:
            message += f" | Context: {filtered_context}"
    
    logger.error(message, exc_info=True)