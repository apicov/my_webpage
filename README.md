# Personal AI Assistant Platform

A modern web application featuring an AI-powered chat assistant built with React frontend and Flask API backend.

## 🚀 Features

- **Interactive Chat Interface**: Real-time conversation with AI assistant
- **Professional Portfolio**: Showcase skills, experience, and personal information
- **Modern UI**: Responsive design with beautiful gradients and animations
- **API-Driven**: Clean separation between frontend and backend
- **CV Download**: Direct access to downloadable resume

## 🛠️ Tech Stack

### Frontend
- **React 18** - Modern component-based UI library
- **React Router** - Client-side routing
- **Tailwind CSS** - Utility-first CSS framework
- **Chart.js** - Data visualization capabilities

### Backend
- **Flask** - Lightweight Python web framework
- **Flask-CORS** - Cross-origin resource sharing
- **AI Assistant** - Custom AI integration for intelligent responses

## 📁 Project Structure

```
my_webpage/
├── app.py                 # Flask API backend
├── frontend/              # React application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   └── services/      # API service layer
│   └── public/            # Static assets
├── data/                  # Personal data and content
├── requirements.txt       # Python dependencies
└── wsgi.py               # Production deployment config
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm

### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start Flask API server
python app.py
```
*Backend runs on http://localhost:5000*

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start React development server
npm start
```
*Frontend runs on http://localhost:3000*

## 🔗 API Endpoints

- `POST /api/chat` - Send messages to AI assistant
- `GET /api/user-info` - Retrieve user profile information

## 🌟 Key Components

- **HeroSection** - Personal introduction and profile display
- **ChatInterface** - Interactive AI chat functionality  
- **SkillsSection** - Technical skills showcase
- **ExperienceSection** - Professional experience timeline

## 🚀 Apache Deployment

### Step 1: Prepare the Application

```bash
# Build React for production
cd frontend
npm run build

# This creates a 'build' folder with optimized static files
```

### Step 2: Install Apache & mod_wsgi

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install apache2 libapache2-mod-wsgi-py3

# Enable mod_wsgi
sudo a2enmod wsgi
sudo a2enmod rewrite
```

### Step 3: Deploy to Apache Directory

```bash
# Copy your project to Apache directory
sudo cp -r /path/to/my_webpage /var/www/html/

# Set permissions
sudo chown -R www-data:www-data /var/www/html/my_webpage
sudo chmod -R 755 /var/www/html/my_webpage
```

### Step 4: Build Frontend for Production

The `api.js` file automatically detects the environment and configures URLs:
- **Development**: `http://localhost:5000/api/*`
- **Production**: Relative URLs (`/api/*`) - works with Apache

```bash
# Build React for production
cd frontend
npm run build
```

**Optional**: Set custom API URL via environment variable:
```bash
# For custom API endpoint
export REACT_APP_API_URL=https://api.yourdomain.com
npm run build
```

### Step 5: Create Apache Virtual Host

Create `/etc/apache2/sites-available/YOUR_DOMAIN.conf`:

```apache
<IfModule mod_ssl.c>
<VirtualHost *:443>
    ServerName YOUR_DOMAIN
    ServerAlias www.YOUR_DOMAIN
    DocumentRoot /var/www/YOUR_DOMAIN/frontend/build

    # Serve React app
    <Directory /var/www/YOUR_DOMAIN/frontend/build>
        Options -Indexes
        AllowOverride All
        Require all granted

        # React Router support
        RewriteEngine On
        RewriteBase /
        RewriteRule ^index\.html$ - [L]
        RewriteCond %{REQUEST_FILENAME} !-f
        RewriteCond %{REQUEST_FILENAME} !-d
        RewriteRule . /index.html [L]
    </Directory>

    # Flask API endpoint
    WSGIDaemonProcess YOUR_DOMAIN python-home=/path/to/your/python/env python-path=/path/to/your/site-packages
    WSGIProcessGroup YOUR_DOMAIN
    WSGIScriptAlias /api /var/www/YOUR_DOMAIN/wsgi.py

    <Directory /var/www/YOUR_DOMAIN>
        WSGIApplicationGroup %{GLOBAL}
        Require all granted
    </Directory>

    ErrorLog ${APACHE_LOG_DIR}/error.log
    CustomLog ${APACHE_LOG_DIR}/access.log combined

Include /etc/letsencrypt/options-ssl-apache.conf
SSLCertificateFile /etc/letsencrypt/live/YOUR_DOMAIN/fullchain.pem
SSLCertificateKeyFile /etc/letsencrypt/live/YOUR_DOMAIN/privkey.pem
</VirtualHost>
</IfModule>
```

**Key Configuration Notes:**
- Replace `YOUR_DOMAIN` with your actual domain name
- Replace `/path/to/your/python/env` with your Python environment path (e.g., `/home/user/miniconda3/envs/myenv`)
- Replace `/path/to/your/site-packages` with your site-packages path (e.g., `/home/user/miniconda3/envs/myenv/lib/python3.10/site-packages`)
- This configuration includes SSL support via Let's Encrypt

### Step 6: Update WSGI Configuration

Your `wsgi.py` is already configured, but ensure it's production-ready:

```python
#!/usr/bin/python3
import sys
import os

# Add your project directory to Python path
sys.path.insert(0, "/var/www/html/my_webpage/")

# Set environment variables
os.environ['FLASK_ENV'] = 'production'

from app import app as application

if __name__ == "__main__":
    application.run()
```

### Step 7: Enable Site & Restart Apache

```bash
# Enable the site
sudo a2ensite my_webpage.conf

# Disable default site (optional)
sudo a2dissite 000-default.conf

# Test Apache configuration
sudo apache2ctl configtest

# Restart Apache
sudo systemctl restart apache2
```

### Step 8: SSL Certificate (Recommended)

```bash
# Install certbot
sudo apt install certbot python3-certbot-apache

# Get SSL certificate
sudo certbot --apache -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Production Environment Variables

Create `/var/www/html/my_webpage/.env`:
```
FLASK_ENV=production
MY_NAME=Your Name
MY_LAST_NAME=Your Last Name
```

### Troubleshooting

1. **Check Apache error logs**: `sudo tail -f /var/log/apache2/error.log`
2. **Verify permissions**: `ls -la /var/www/html/my_webpage`
3. **Test WSGI**: `python3 /var/www/html/my_webpage/wsgi.py`
4. **Check Apache status**: `sudo systemctl status apache2`

## 📝 Configuration

Create a `.env` file in the root directory:
```
MY_NAME=Your Name
MY_LAST_NAME=Your Last Name
```

Update `data/personal_info.json` with your information:
```json
{
  "name": "Your Name",
  "title": "Your Title", 
  "bio": "Your bio",
  "skills": ["Skill 1", "Skill 2"],
  "experience": [...]
}
```

---

Built with ❤️ using React and Flask
