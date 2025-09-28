# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (Flask API)
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start Flask development server
python app.py
```
Backend runs on http://localhost:5000

### Frontend (React + Vite)
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Type checking
npm run typecheck

# Linting
npm run lint

# Preview production build
npm run preview
```
Frontend runs on http://localhost:3000

## Architecture Overview

### Core Structure
- **Full-stack web application** with React frontend and Flask backend
- **Multi-agent AI system** powered by LangGraph for intelligent conversation routing
- **Dual routing**: Both `/api/*` and direct `/*` paths supported for development/production compatibility

### Key Components

#### Backend (`app.py`)
- Flask API server with CORS enabled
- LangGraph orchestrator for AI agent routing (`agents/orchestrator.py`)
- Personal data loaded from `data/` directory (JSON, markdown files)
- Environment variables from `.env` file

#### Frontend (`frontend/src/`)
- React 19 with TypeScript 5.7
- Vite 6.0 build system
- React Router v7 for routing
- Chart.js for data visualization
- Tailwind CSS (configured in `index.css`)

#### AI Agent System (`agents/`)
- **LangGraphOrchestrator**: Semantic routing between specialized agents
- **ChatAgent**: Main conversational agent with personal context
- **BaseAgent**: Abstract base class for all agents
- Uses LangChain + Groq for LLM integration

### Project Structure
```
├── app.py                    # Flask API server
├── agents/                   # LangGraph multi-agent system
│   ├── orchestrator.py       # Main agent router
│   ├── chat_agent.py         # Personal chat agent
│   └── base_agent.py         # Agent base class
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/       # React components (.tsx)
│   │   ├── pages/           # Page components
│   │   ├── services/        # API layer
│   │   └── types/           # TypeScript definitions
│   └── package.json         # Frontend dependencies
├── data/                    # Personal content
│   ├── personal_info.json   # Profile data
│   ├── summary.txt          # Personal summary
│   └── resume.md           # Resume content
└── requirements.txt         # Python dependencies
```

## API Endpoints
- `POST /chat` or `/api/chat` - AI chat interface
- `GET /user-info` or `/api/user-info` - Personal profile data

## Development Notes

### Agent System
- Agents are semantically routed via LLM reasoning (no keyword matching)
- See `AGENTS_ARCHITECTURE.md` for detailed multi-agent documentation
- New agents can be added by extending `BaseAgent` and registering in orchestrator

### Environment Setup
- Create `.env` file with `MY_NAME` and `MY_LAST_NAME` variables
- Update `data/personal_info.json` with personal information
- Frontend automatically detects development vs production API URLs

### Testing & Quality
- Run `npm run typecheck` for TypeScript validation
- Run `npm run lint` for code quality checks
- No specific test framework detected - check with user for testing approach

### Deployment
- Production build: `npm run build` in frontend directory
- Apache deployment configuration included in README.md
- WSGI configuration in `wsgi.py` for production deployment
- Do not be flunky. respond honestly to questions and opinions
- Dont say "You're absolutely right" all the time. it is annoying. Be you