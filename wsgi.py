import sys
import os

sys.path.insert(0, '/var/www/apicov.xyz')

# Set the Flask app environment variable
os.environ['FLASK_APP'] = 'app.py'

from app import app as application  # Import the Flask app