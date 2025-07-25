# 🚀 Production Deployment Guide

This guide helps you deploy your React + Flask application to Apache safely and securely.

## 📋 **Pre-Deployment Checklist**

### **1. Build React App**
```bash
cd frontend
npm run build
```

### **2. Upload Files to Server**
Upload these directories to your server:
- `frontend/build/` → `/var/www/YOUR_DOMAIN/frontend/build/`
- `app.py`, `wsgi.py`, `requirements.txt` → `/var/www/YOUR_DOMAIN/`
- `data/` directory → `/var/www/YOUR_DOMAIN/data/`

### **3. Install Python Dependencies**
```bash
cd /var/www/YOUR_DOMAIN
pip install -r requirements.txt
```

## 🔧 **Apache Configuration**

### **1. Create Virtual Host**
1. Copy `deployment/apache-template.conf` to your server
2. Replace `YOUR_DOMAIN` with your actual domain
3. Save as `/etc/apache2/sites-available/YOUR_DOMAIN.conf`

### **2. Enable Site**
```bash
sudo a2ensite YOUR_DOMAIN.conf
sudo a2enmod wsgi rewrite ssl
sudo systemctl restart apache2
```

### **3. Set Permissions**
```bash
sudo chown -R www-data:www-data /var/www/YOUR_DOMAIN
sudo chmod -R 755 /var/www/YOUR_DOMAIN
```

## 🔒 **Security Best Practices**

### **Configuration Files**
- ❌ **Never commit actual Apache configs** with real domains/paths
- ✅ **Use templates** with placeholders (like this repo)
- ✅ **Store actual configs** outside version control

### **Environment Variables**
Create `/var/www/YOUR_DOMAIN/.env`:
```
FLASK_ENV=production
MY_NAME=Your Name
MY_LAST_NAME=Your Last Name
```

### **File Permissions**
```bash
# Web files - readable by Apache
sudo chmod 644 /var/www/YOUR_DOMAIN/frontend/build/*

# Python files - executable by Apache
sudo chmod 755 /var/www/YOUR_DOMAIN/*.py

# Data files - readable only
sudo chmod 644 /var/www/YOUR_DOMAIN/data/*

# Configuration files - secure
sudo chmod 600 /var/www/YOUR_DOMAIN/.env
```

## 🧪 **Testing Deployment**

### **Test URLs**
- `https://YOUR_DOMAIN/` → React app should load
- `https://YOUR_DOMAIN/api/user-info` → JSON response
- `https://YOUR_DOMAIN/api/chat` → POST endpoint (test with curl)

### **Troubleshooting**
1. **Check Apache error logs**: `sudo tail -f /var/log/apache2/error.log`
2. **Verify file paths**: Ensure all files exist where Apache expects them
3. **Test permissions**: Make sure www-data can read your files
4. **Check SSL**: Verify SSL certificates are valid

## 📞 **Common Issues**

### **404 Not Found**
- Check DocumentRoot path in Apache config
- Verify React build files exist
- Ensure Apache can read the files

### **500 Internal Server Error**
- Check Apache error logs
- Verify Python dependencies installed
- Check WSGI configuration
- Ensure proper file permissions

### **API Not Working**
- Verify Flask app runs locally first
- Check WSGI path configuration
- Test API endpoints directly
- Check CORS settings

## 🔄 **Update Process**

When updating your application:

1. **Build new React version**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Upload new files**:
   ```bash
   rsync -av frontend/build/ user@server:/var/www/YOUR_DOMAIN/frontend/build/
   rsync -av *.py data/ user@server:/var/www/YOUR_DOMAIN/
   ```

3. **Restart Apache** (if needed):
   ```bash
   sudo systemctl reload apache2
   ```

---

## 🛡️ **Security Notes**

- Keep your actual Apache configs **outside** this repository
- Use environment variables for sensitive data
- Regularly update dependencies and SSL certificates
- Monitor Apache logs for security issues
- Use fail2ban or similar tools for additional protection 