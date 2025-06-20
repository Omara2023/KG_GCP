import logging

def setup_logging():
    """Configures the root logger for the application."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

if __name__ == '__main__':
    setup_logging()
    test_logger = logging.getLogger(__name__)
    test_logger.debug("This is a debug message from logging_config.py")
    test_logger.info("This is an info message from logging_config.py")
    test_logger.warning("This is a warning message from logging_config.py")
    test_logger.error("This is an error message from logging_config.py")