import logging
import os

# Flag to ensure the logging setup is done only once
_logging_configured = False

def setup_logging():
    global _logging_configured
    if not _logging_configured:
        log_level = logging.DEBUG if os.getenv('ENV') == 'dev' else logging.INFO
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        _logging_configured = True