import sys
import os

# Add the Conda environment's site-packages to the Python path
conda_env_path = '/home/pico/miniconda3/envs/lino'
sys.path.insert(0, os.path.join(conda_env_path, 'lib/python3.10/site-packages'))

# Add the path to your application
sys.path.insert(0, '/var/www/apicov.xyz')

# Set the Flask app environment variable
os.environ['FLASK_APP'] = 'app.py'

# Import the Flask app
try:
    from app import app as application  # Adjust this line based on your app structure
except ImportError as e:
    print(f"Error importing app: {e}")
    raise