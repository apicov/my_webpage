# Adding Projects to Your Portfolio

This guide explains how to add new projects to your portfolio website.

## Project Structure

Projects are stored as markdown files in the `/data/projects/` directory. Each project is a separate `.md` file with frontmatter metadata and markdown content.

## Creating a New Project

### 1. Create a New Markdown File

Create a new file in `/data/projects/` with a descriptive filename:
```
/data/projects/your-project-name.md
```

### 2. Add Frontmatter Metadata

Start your file with frontmatter (between `---` markers) containing project metadata:

```markdown
---
id: unique-project-id
title: Your Project Title
description: A brief one-line description of your project
technologies: React, Node.js, MongoDB, Docker
status: completed
featured: true
githubUrl: https://github.com/yourusername/project
liveUrl: https://your-project.com
demoUrl: https://youtube.com/watch?v=demo
startDate: Jan 2024
endDate: Mar 2024
thumbnail: /images/projects/project-thumbnail.jpg
---
```

#### Frontmatter Fields:

- **id** (required): Unique identifier for the project
- **title** (required): Project name
- **description** (required): Short description for the project card
- **technologies** (required): Comma-separated list of technologies used
- **status** (required): One of: `completed`, `in-progress`, or `planned`
- **featured** (optional): Set to `true` to highlight this project
- **githubUrl** (optional): Link to GitHub repository
- **liveUrl** (optional): Link to live deployment
- **demoUrl** (optional): Link to demo video
- **startDate** (optional): When you started the project
- **endDate** (optional): When you completed the project
- **thumbnail** (optional): Path to thumbnail image for project card

### 3. Add Project Content

After the frontmatter, add your project content using markdown:

```markdown
## Overview

Describe your project in detail here. What problem does it solve? What was your approach?

![Main Interface](/images/projects/your-project-main.png)

## Key Features

### Feature 1
Description of the feature with details.

![Feature Screenshot](/images/projects/feature1.png)

- Bullet point 1
- Bullet point 2
- Bullet point 3

### Feature 2
Another feature description.

## Technical Implementation

Explain the technical details, architecture, challenges, and solutions.

![Architecture Diagram](/images/projects/architecture.svg)

## Results

Share metrics, improvements, or impact of your project.
```

## Adding Images

### 1. Store Images

Place your project images in `/frontend/public/images/projects/`:
```
/frontend/public/images/projects/
├── project1-thumbnail.jpg
├── project1-dashboard.png
├── project1-mobile.jpg
└── project1-architecture.svg
```

### 2. Reference Images in Markdown

In your markdown content, reference images using:
```markdown
![Alt text](/images/projects/your-image.png)
```

### Image Best Practices:

- **Thumbnails**: 800x600px recommended, will be cropped to fit card
- **Content images**: Max width 1920px for optimal loading
- **File formats**: Use `.jpg` for photos, `.png` for screenshots, `.svg` for diagrams
- **File size**: Optimize images to be under 500KB when possible

## Complete Example

Here's a complete example project file:

```markdown
---
id: ai-chatbot
title: AI Customer Service Chatbot
description: Intelligent chatbot with natural language processing for customer support
technologies: Python, TensorFlow, React, FastAPI, PostgreSQL
status: completed
featured: true
githubUrl: https://github.com/yourusername/ai-chatbot
liveUrl: https://chatbot-demo.com
startDate: Jan 2024
endDate: Mar 2024
thumbnail: /images/projects/chatbot-thumbnail.jpg
---

## Overview

This AI-powered chatbot revolutionizes customer service by providing instant, accurate responses to customer inquiries 24/7.

![Chatbot Interface](/images/projects/chatbot-main.png)

## Key Features

### Natural Language Understanding

The chatbot uses advanced NLP to understand context and intent:

![NLP Processing](/images/projects/chatbot-nlp.png)

- Supports 95+ languages
- Context-aware conversations
- Sentiment analysis
- Intent classification

### Smart Routing

Automatically routes complex queries to human agents:

![Routing Dashboard](/images/projects/chatbot-routing.png)

## Technical Architecture

Built with a microservices architecture for scalability:

![Architecture](/images/projects/chatbot-architecture.svg)

### Backend
- **FastAPI** for high-performance API
- **TensorFlow** for ML models
- **PostgreSQL** for conversation history
- **Redis** for session management

### Frontend
- **React** with TypeScript
- **WebSocket** for real-time messaging
- **Tailwind CSS** for responsive design

## Results

- **60% reduction** in response time
- **40% decrease** in support tickets
- **4.8/5 customer satisfaction** rating
- Processing **10,000+ conversations** daily
```

## Tips for Great Project Documentation

1. **Start with impact**: Lead with what the project achieves
2. **Show, don't just tell**: Include screenshots and diagrams
3. **Explain technical choices**: Why did you choose certain technologies?
4. **Include challenges**: What problems did you solve?
5. **Quantify results**: Use metrics when possible
6. **Keep it scannable**: Use headers, bullets, and short paragraphs

## Viewing Your Projects

After adding a project:

1. Restart your Flask backend (if running)
2. Your project will appear automatically in the Projects section
3. Click on the project card to view the full details
4. Featured projects appear at the top of the list

## Troubleshooting

### Project not appearing?
- Check that the file is in `/data/projects/`
- Ensure the file has `.md` extension
- Verify the frontmatter syntax is correct
- Restart the Flask backend

### Images not showing?
- Verify images are in `/frontend/public/images/projects/`
- Check the image path starts with `/images/projects/`
- Ensure the image filename matches exactly (case-sensitive)

### Markdown not rendering correctly?
- The project page supports GitHub Flavored Markdown
- Use proper markdown syntax for headers, lists, etc.
- Test with a markdown preview tool if needed