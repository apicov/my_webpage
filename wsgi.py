import sys
import os

# Activate the Conda environment
activate_this = '/home/pico/miniconda3/envs/lino/bin/activate_this.py'
exec(open(activate_this).read(), dict(__file__=activate_this))

# Add the path to your application
sys.path.insert(0, '/var/www/apicov.xyz')

# Set the Flask app environment variable
os.environ['FLASK_APP'] = 'app.py'

from app import app as application  # Import the Flask app