import React from 'react';
import PropTypes from 'prop-types';

function HeroSection({ userInfo }) {
  const scrollToChat = () => {
    const chatSection = document.getElementById('chat-section');
    if (chatSection) {
      chatSection.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }
  };

  return (
    <section className="w-full h-full flex items-center">
      <div className="gradient-bg text-white w-full h-full flex items-center rounded-xl p-6 lg:p-10">
        <div className="flex flex-col lg:flex-row items-center w-full gap-8">
          {/* Profile Info */}
          <div className="w-full lg:w-2/3 flex flex-col justify-center items-start space-y-4 lg:space-y-6 text-left">
            <h1 className="text-4xl lg:text-5xl font-bold mb-2 lg:mb-3 leading-tight text-left">
              Hi, I'm {userInfo?.name || 'Your Name'}
            </h1>
            <h2 className="text-xl lg:text-2xl mb-2 opacity-90 font-semibold text-left">
              {userInfo?.title || 'Software Engineer'}
            </h2>
            <p className="text-base lg:text-lg leading-relaxed mb-2 opacity-90 max-w-xl text-left whitespace-pre-line">
              {userInfo?.bio || 'Passionate about technology and innovation.'}
            </p>
            <div className="flex flex-wrap gap-3 mt-2">
              <a href="/static/cv.pdf" download style={{ display: 'inline-block' }}>
                <button className="bg-white text-purple-600 px-4 py-3 md:px-5 md:py-2 rounded-lg font-semibold hover:bg-gray-100 transition-colors text-sm lg:text-base flex items-center min-h-[44px]">
                  <i className="fas fa-download mr-2"></i>Download CV
                </button>
              </a>
              <button 
                onClick={scrollToChat}
                className="border-2 border-white text-white px-4 py-3 md:px-5 md:py-2 rounded-lg font-semibold hover:bg-white hover:text-purple-600 transition-colors text-sm lg:text-base min-h-[44px]"
              >
                <i className="fas fa-comments mr-2"></i>Ask My AI Assistant
              </button>
            </div>
          </div>
          
          {/* Profile Picture */}
          <div className="w-full lg:w-1/3 flex justify-center items-center mt-6 lg:mt-0">
            <div className="w-32 h-32 sm:w-40 sm:h-40 lg:w-56 lg:h-56 rounded-full overflow-hidden border-4 border-white shadow-2xl flex-shrink-0">
              <img 
                src="/static/myphoto.jpg" 
                alt={userInfo?.name || 'Profile'} 
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.target.src = 'https://via.placeholder.com/400x400/4F46E5/FFFFFF?text=Your+Photo';
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

HeroSection.propTypes = {
  userInfo: PropTypes.shape({
    name: PropTypes.string,
    title: PropTypes.string,
    bio: PropTypes.string
  })
};

export default HeroSection; 