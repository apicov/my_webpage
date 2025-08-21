# React TypeScript Frontend

Modern React frontend built with TypeScript, Vite, and Tailwind CSS for the portfolio website with AI assistant capabilities.

## 🚀 Features

- **TypeScript**: Full type safety and better developer experience
- **Vite**: Lightning-fast HMR and builds
- **React 19**: Latest React features and performance improvements
- **Portfolio Display**: Professional profile with skills and experience
- **AI Chat Interface**: Interactive chat with AI assistant
- **Responsive Design**: Mobile-friendly interface
- **Modern UI**: Built with Tailwind CSS

## 🛠️ Tech Stack

- **React 19.0** - UI library
- **TypeScript 5.7** - Type-safe JavaScript
- **Vite 6.0** - Build tool
- **React Router v7** - Routing
- **Tailwind CSS** - Styling
- **Chart.js** - Data visualization

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/         # React components (.tsx)
│   │   ├── HeroSection.tsx
│   │   ├── ChatInterface.tsx
│   │   ├── SkillsSection.tsx
│   │   └── ExperienceSection.tsx
│   ├── pages/              # Page components (.tsx)
│   │   └── HomePage.tsx
│   ├── services/           # API services (.ts)
│   │   └── api.ts
│   ├── types/              # TypeScript definitions
│   │   └── index.ts
│   ├── App.tsx             # Main App component
│   ├── index.tsx           # Entry point
│   └── index.css           # Global styles
├── index.html              # Entry HTML
├── vite.config.ts          # Vite configuration
├── tsconfig.json           # TypeScript configuration
└── package.json            # Dependencies and scripts
```

## 🏃 Quick Start

### Prerequisites

- Node.js 18+ 
- npm 9+
- Flask backend running on port 5000

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will run on `http://localhost:3000`

## 📜 Available Scripts

```bash
npm run dev        # Start development server with HMR
npm run build      # Build for production
npm run preview    # Preview production build locally
npm run typecheck  # Run TypeScript type checking
npm run lint       # Run ESLint
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file for local development:

```env
# Optional: Override API URL (defaults to proxy in dev, /api in production)
VITE_API_URL=http://localhost:5000
```

### Vite Proxy Configuration

The development server proxies API requests to the Flask backend:

```typescript
// vite.config.ts
proxy: {
  '/api': {
    target: 'http://localhost:5000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ''),
  }
}
```

## 🏗️ Building for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

This creates a `build` folder with optimized production files.

## 🚀 Deployment

### Static Hosting (Netlify, Vercel, etc.)

1. Build the project: `npm run build`
2. Deploy the `build` folder
3. Configure environment variables if needed

### Apache/Nginx

1. Build: `npm run build`
2. Serve the `build` folder
3. Configure proxy for `/api` routes to Flask backend

Example Nginx config:
```nginx
location / {
    root /path/to/build;
    try_files $uri /index.html;
}

location /api/ {
    proxy_pass http://localhost:5000/;
}
```

## 🎨 Styling

- **Tailwind CSS**: Utility-first CSS framework
- **Custom styles**: Add to `src/index.css`
- **Component styles**: Use Tailwind classes directly in components

## 🔌 API Integration

The frontend communicates with Flask through typed API services:

```typescript
// src/services/api.ts
export const getUserInfo = async (): Promise<UserInfo> => {
  // Typed API call
}

export const chatWithAI = async (messages: ChatMessage[]): Promise<Response> => {
  // Typed chat API
}
```

## 🐛 Troubleshooting

### Common Issues

- **TypeScript errors**: Run `npm run typecheck` to identify issues
- **Build errors**: Ensure Node.js 18+ and correct dependencies
- **API connection**: Verify Flask backend is running on port 5000
- **CORS issues**: Check Flask CORS configuration

### Development Tips

- Use `npm run typecheck` before committing
- VSCode users: Install TypeScript + ESLint extensions
- Check browser console for runtime errors
- Use React DevTools for component debugging

## 📚 Type Definitions

Core types are defined in `src/types/index.ts`:

```typescript
interface UserInfo {
  name?: string;
  title?: string;
  bio?: string;
  skills?: string[];
  experience?: Experience[];
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  media?: MediaContent;
}
```

## 🤝 Contributing

1. Follow TypeScript best practices
2. Maintain type safety (avoid `any`)
3. Use functional components with hooks
4. Follow existing code structure
5. Test thoroughly before committing

---

Built with ⚡ Vite + 🔷 TypeScript + ⚛️ React