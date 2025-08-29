import React, { useState, useEffect } from 'react';
import HeroSection from '../components/HeroSection';
import MultiModalChatInterface from '../components/MultiModalChatInterface';
import SkillsSection from '../components/SkillsSection';
import ExperienceSection from '../components/ExperienceSection';
import ProjectsSection from '../components/ProjectsSection';
import { getUserInfo, getProjects } from '../services/api';
import { UserInfo, Project } from '../types';

const HomePage: React.FC = () => {
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNav, setShowNav] = useState(false);

  // Handle scroll to show/hide navigation
  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY;
      const heroHeight = window.innerHeight * 0.3; // Show nav after scrolling past 30% of viewport
      
      // Hide nav when user is in chat area (mobile only)
      const chatSection = document.getElementById('chat-section');
      if (chatSection && window.innerWidth <= 768) {
        const chatRect = chatSection.getBoundingClientRect();
        const isInChatArea = chatRect.top < window.innerHeight * 0.5 && chatRect.bottom > 0;
        setShowNav(scrollPosition > heroHeight && !isInChatArea);
      } else {
        // Desktop: keep normal behavior
        setShowNav(scrollPosition > heroHeight);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Smooth scroll to section
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      const offsetTop = element.offsetTop - 80; // Account for fixed nav
      window.scrollTo({ top: offsetTop, behavior: 'smooth' });
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch user info and projects in parallel
        const [info, projectsList] = await Promise.all([
          getUserInfo(),
          getProjects()
        ]);
        setUserInfo(info);
        setProjects(projectsList);
      } catch (error) {
        console.error('Failed to fetch data:', error);
        // Fallback to default info if API fails
        setUserInfo({
          name: 'Your Name',
          title: 'Software Engineer',
          bio: 'Passionate about technology and innovation.',
          skills: ['Python', 'JavaScript', 'React', 'Flask', 'IoT'],
          experience: [
            {
              role: 'Software Engineer',
              company: 'Tech Company',
              period: '2020 - Present',
              description: 'Developing innovative solutions.'
            }
          ]
        });
        setProjects([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="loading-spinner"></div>
        <span className="ml-2">Loading...</span>
      </div>
    );
  }

  return (
    <div className="bg-gray-50">
      {/* Sticky Navigation */}
      {showNav && (
        <nav className="fixed top-4 left-1/2 transform -translate-x-1/2 z-[9999] bg-white border border-gray-200 rounded-full shadow-xl px-4 py-2">
          <div className="flex gap-1 text-sm font-medium">
            <button
              onClick={() => scrollToSection('hero')}
              className="px-3 py-1 rounded-full hover:bg-blue-100 hover:text-blue-600 transition-colors text-gray-700"
            >
              About
            </button>
            <button
              onClick={() => scrollToSection('skills')}
              className="px-3 py-1 rounded-full hover:bg-blue-100 hover:text-blue-600 transition-colors text-gray-700"
            >
              Skills
            </button>
            <button
              onClick={() => scrollToSection('projects')}
              className="px-3 py-1 rounded-full hover:bg-blue-100 hover:text-blue-600 transition-colors text-gray-700"
            >
              Projects
            </button>
            <button
              onClick={() => scrollToSection('experience')}
              className="px-3 py-1 rounded-full hover:bg-blue-100 hover:text-blue-600 transition-colors text-gray-700"
            >
              Experience
            </button>
          </div>
        </nav>
      )}
      <div id="hero" className="flex flex-col lg:flex-row container mx-auto px-4 py-4 sm:py-8 gap-4 sm:gap-8 lg:items-stretch">
        {/* Left Side: Hero Section */}
        <div className="w-full lg:w-1/2 mb-4 sm:mb-8 lg:mb-0 flex items-center justify-center">
          <HeroSection userInfo={userInfo || undefined} />
        </div>
        
        {/* Right Side: Multi-Modal Chat Interface */}
        <div id="chat-section" className="w-full lg:w-1/2 flex items-center justify-center">
          <MultiModalChatInterface userInfo={userInfo || undefined} />
        </div>
      </div>

      {/* Skills Section */}
      <div id="skills">
        <SkillsSection skills={userInfo?.skills || []} />
      </div>
      
      {/* Projects Section */}
      <div id="projects">
        <ProjectsSection projects={projects} />
      </div>
      
      {/* Experience Section */}
      <div id="experience">
        <ExperienceSection experience={userInfo?.experience || []} />
      </div>
      
      {/* Footer */}
      <footer className="gradient-bg text-white py-8">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; {new Date().getFullYear()} {userInfo?.name || 'Your Name'}. Built with React & Flask</p>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;