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
    # Path to your Anaconda environment's bin directory
    activate_path = '/home/pico/miniconda3/envs/lino/bin'
    python_path = os.path.join(activate_path, 'python')
    sys.executable = python_path

    # Add site-packages path
    site_packages_path = '/home/pico/miniconda3/envs/lino/lib/python3.10/site-packages'
    sys.path.insert(0, site_packages_path)

    # Add project directory to Python path
    project_dir = '/var/www/apicov.xyz'
    sys.path.insert(0, project_dir)

    # Add agents and routes directories to Python path for module imports
    agents_dir = os.path.join(project_dir, 'agents')
    routes_dir = os.path.join(project_dir, 'routes')
    sys.path.insert(0, agents_dir)
    sys.path.insert(0, routes_dir)

    # CHANGE WORKING DIRECTORY - Add this line
    os.chdir(project_dir)  # Change to where your .env file is located
    
    # Log system information
    logging.debug(f"Python Executable: {sys.executable}")
    logging.debug(f"Python Version: {sys.version}")
    logging.debug(f"Project Directory: {project_dir}")
    logging.debug(f"Agents Directory: {agents_dir}")
    logging.debug(f"Routes Directory: {routes_dir}")
    logging.debug("Python Path Entries:")
    for path in sys.path:
        logging.debug(path)

    # Test critical imports before importing app
    try:
        import pydantic
        logging.debug("✓ pydantic imported successfully")
    except ImportError as e:
        logging.error(f"✗ pydantic import failed: {e}")

    try:
        import langchain_groq
        logging.debug("✓ langchain_groq imported successfully")
    except ImportError as e:
        logging.error(f"✗ langchain_groq import failed: {e}")

    try:
        import langchain_core
        logging.debug("✓ langchain_core imported successfully")
    except ImportError as e:
        logging.error(f"✗ langchain_core import failed: {e}")

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