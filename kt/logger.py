import logging
import threading

class KTLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:  # Ensure thread-safety
                if cls._instance is None:
                    cls._instance = super(KTLogger, cls).__new__(cls)
                    cls._instance._initialize_logger()
        return cls._instance
    def _initialize_logger(self):
        """
        Initialize the logger with basic settings.
        """
        self.logger = logging.getLogger('KTLogger')
        self.logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        Return the logger instance.
        """
        return self.logger

# Usage example:
if __name__ == "__main__":
    logger = KTLogger().get_logger()

    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
