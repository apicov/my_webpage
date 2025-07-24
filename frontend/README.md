# React Frontend for My Webpage

This is the React frontend for the portfolio website with AI assistant and IoT integration capabilities.

## Features

- **Portfolio Display**: Professional profile with skills and experience
- **AI Chat Interface**: Interactive chat with AI assistant
- **Real-time Communication**: WebSocket support for IoT devices
- **Responsive Design**: Mobile-friendly interface
- **Modern UI**: Built with Tailwind CSS

## Setup

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Flask backend running on port 5000

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The React app will run on `http://localhost:3000`

### Building for Production

```bash
npm run build
```

This creates a `build` folder with optimized production files.

## Project Structure

```
frontend/
├── public/
│   ├── index.html          # Main HTML file
│   └── manifest.json       # Web app manifest
├── src/
│   ├── components/         # Reusable React components
│   │   ├── HeroSection.js
│   │   ├── ChatInterface.js
│   │   ├── SkillsSection.js
│   │   └── ExperienceSection.js
│   ├── pages/              # Page components
│   │   └── HomePage.js
│   ├── services/           # API services
│   │   └── api.js
│   ├── App.js              # Main App component
│   ├── index.js            # Entry point
│   └── index.css           # Global styles
└── package.json            # Dependencies and scripts
```

## API Integration

The frontend communicates with the Flask backend through the API service in `src/services/api.js`. The backend should be running on `http://localhost:5000`.

## Development

### Adding New Components

1. Create new components in `src/components/`
2. Import and use them in pages or other components
3. Follow the existing component structure and styling patterns

### Adding IoT Features

1. Create new IoT-specific components in `src/components/`
2. Add corresponding API endpoints in `src/services/api.js`
3. Implement real-time features using WebSocket connections

### Styling

The project uses Tailwind CSS for styling. Custom styles can be added to `src/index.css`.

## Deployment

1. Build the production version: `npm run build`
2. Serve the `build` folder with your web server
3. Ensure the Flask backend is accessible from the frontend

## Troubleshooting

- **CORS Issues**: Make sure the Flask backend has CORS enabled
- **API Connection**: Verify the backend is running on port 5000
- **Build Errors**: Check Node.js version and dependencies 