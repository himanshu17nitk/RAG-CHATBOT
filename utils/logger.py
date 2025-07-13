import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys

class APILogger:
    def __init__(self, name="rag_api"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup different log handlers for different purposes"""
        
        # Console handler for real-time debugging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Debug file handler (all logs)
        debug_handler = RotatingFileHandler(
            os.path.join(self.logs_dir, "debug.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        debug_handler.setFormatter(debug_formatter)
        self.logger.addHandler(debug_handler)
        
        # Error file handler (errors only)
        error_handler = RotatingFileHandler(
            os.path.join(self.logs_dir, "errors.log"),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n'
            'Exception: %(exc_info)s\n'
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
        
        # API requests handler
        api_handler = RotatingFileHandler(
            os.path.join(self.logs_dir, "api_requests.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        api_handler.setLevel(logging.INFO)
        api_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        api_handler.setFormatter(api_formatter)
        self.logger.addHandler(api_handler)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message, exc_info=None):
        """Log error message with optional exception info"""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message, exc_info=None):
        """Log critical message with optional exception info"""
        self.logger.critical(message, exc_info=exc_info)
    
    def log_api_request(self, method, endpoint, user_id=None, session_id=None, **kwargs):
        """Log API request details"""
        log_message = f"API Request: {method} {endpoint}"
        if user_id:
            log_message += f" | User: {user_id}"
        if session_id:
            log_message += f" | Session: {session_id}"
        if kwargs:
            log_message += f" | Params: {kwargs}"
        self.info(log_message)
    
    def log_api_response(self, method, endpoint, status_code, response_time=None, **kwargs):
        """Log API response details"""
        log_message = f"API Response: {method} {endpoint} | Status: {status_code}"
        if response_time:
            log_message += f" | Time: {response_time:.3f}s"
        if kwargs:
            log_message += f" | Details: {kwargs}"
        self.info(log_message)
    
    def log_training_step(self, step, user_id, filename, **kwargs):
        """Log training process steps"""
        log_message = f"Training Step: {step} | User: {user_id} | File: {filename}"
        if kwargs:
            log_message += f" | Details: {kwargs}"
        self.info(log_message)
    
    def log_prediction_step(self, step, user_id, session_id, query_length=None, **kwargs):
        """Log prediction process steps"""
        log_message = f"Prediction Step: {step} | User: {user_id} | Session: {session_id}"
        if query_length:
            log_message += f" | Query Length: {query_length}"
        if kwargs:
            log_message += f" | Details: {kwargs}"
        self.info(log_message)

# Global logger instance
api_logger = APILogger() 
