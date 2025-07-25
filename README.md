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

## 🚀 Deployment

The application is configured for deployment with:
- `wsgi.py` for production WSGI servers
- React build process for static file generation
- Environment variable support via `.env`

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
