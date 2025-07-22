# My Website & AI Assistant

This project is a personal portfolio website featuring an interactive AI assistant. Built with Flask and JavaScript, it showcases a professional background, skills, and experience, and allows visitors to chat with an AI agent and download a CV.

---

## Features

- Modern, responsive portfolio site
- AI-powered chat assistant (OpenAI API)
- Downloadable CV
- Dynamic, easily updatable personal info (via JSON)
- Media upload and visualization options

---

## Tech Stack

- Python (Flask)
- JavaScript
- Tailwind CSS
- OpenAI API

---

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your personal info**
   - Create a file at `data/personal_info.json` (see structure below).

4. **Add your CV**
   - Place your CV file (e.g., `cv.pdf`) in the `static/` directory.

5. **Run the app**
   ```bash
   python app.py
   ```
   The site will be available at [http://localhost:5000](http://localhost:5000).

---

## Personal Info JSON Structure

Create a file at `data/personal_info.json` with the following structure (example with made-up data):

```json
{
  "name": "Alex Quantum",
  "title": "Space Robotics Engineer",
  "bio": "Inventor, dreamer, and robotics enthusiast. Alex has designed robots for Mars exploration, underwater archaeology, and even competitive pizza delivery. Passionate about AI, space, and building things that move.",
  "skills": [
    "Space Robotics",
    "AI Navigation",
    "Python",
    "C++",
    "Rocket Propulsion",
    "Pizza Delivery Optimization"
  ],
  "experience": [
    {
      "role": "Lead Robotics Engineer",
      "company": "Galactic Explorers Inc.",
      "period": "2025 - Present",
      "description": "Designed and led the deployment of autonomous robots for asteroid mining and lunar construction."
    },
    {
      "role": "AI Pizza Delivery Specialist",
      "company": "RoboPizza",
      "period": "2023 - 2025",
      "description": "Developed AI algorithms for the world’s first pizza delivery drone fleet."
    }
  ],
  "education": [
    {
      "degree": "PhD in Interplanetary Robotics",
      "school": "Mars Institute of Technology",
      "year": "2023"
    },
    {
      "degree": "BSc in Mechanical Engineering",
      "school": "Atlantis University",
      "year": "2020"
    }
  ]
}
```