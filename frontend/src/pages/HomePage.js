import React, { useState, useEffect } from 'react';
import HeroSection from '../components/HeroSection';
import ChatInterface from '../components/ChatInterface';
import SkillsSection from '../components/SkillsSection';
import ExperienceSection from '../components/ExperienceSection';
import { getUserInfo } from '../services/api';

function HomePage() {
  const [userInfo, setUserInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const info = await getUserInfo();
        setUserInfo(info);
      } catch (error) {
        console.error('Failed to fetch user info:', error);
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
      } finally {
        setLoading(false);
      }
    };

    fetchUserInfo();
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
      <div className="flex flex-col lg:flex-row container mx-auto px-4 py-8 gap-8">
        {/* Left Side: Hero Section */}
        <div className="w-full lg:w-1/2 mb-8 lg:mb-0 flex items-center justify-center">
          <HeroSection userInfo={userInfo} />
        </div>
        
        {/* Right Side: Chat Interface */}
        <div className="w-full lg:w-1/2 flex items-center justify-center">
          <ChatInterface userInfo={userInfo} />
        </div>
      </div>

      {/* Skills Section */}
      <SkillsSection skills={userInfo?.skills || []} />
      
      {/* Experience Section */}
      <ExperienceSection experience={userInfo?.experience || []} />
      
      {/* Footer */}
      <footer className="gradient-bg text-white py-8">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; {new Date().getFullYear()} {userInfo?.name || 'Your Name'}. Built with React & Flask</p>
        </div>
      </footer>
    </div>
  );
}

export default HomePage; 