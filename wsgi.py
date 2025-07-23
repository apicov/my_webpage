import sys
import os
import logging

# Configure logging BEFORE importing the app
logging.basicConfig(
    filename='/var/www/apicov.xyz/wsgi_debug.log',  # Log file path
    level=logging.DEBUG,  # Set logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

try:
    # Add Conda environment paths
    conda_env_path = '/home/pico/miniconda3/envs/lino'
    sys.path.insert(0, os.path.join(conda_env_path, 'lib/python3.10/site-packages'))
    
    # Log system information
    logging.debug(f"Python Executable: {sys.executable}")
    logging.debug(f"Python Version: {sys.version}")
    logging.debug("Python Path Entries:")
    for path in sys.path:
        logging.debug(path)

    # Redirect print to logging
    def print_to_log(*args, **kwargs):
        message = ' '.join(map(str, args))
        logging.debug(message)
    
    # Replace built-in print
    __builtins__['print'] = print_to_log

    # Import the application
    from app import app as application
    logging.debug("Application imported successfully")

except Exception as e:
    # Detailed error logging
    logging.error("Critical error in WSGI script:")
    logging.error(str(e))
    import traceback
    logging.error(traceback.format_exc())
    raise