import React from 'react';

function SkillsSection({ skills }) {
  return (
    <section className="py-16 bg-white">
      <div className="container mx-auto px-4">
        <h3 className="text-3xl font-bold text-center mb-12 text-gray-800">Technical Skills</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {skills.map((skill, index) => (
            <div 
              key={index}
              className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg text-center border border-gray-200 hover:shadow-md transition-shadow"
            >
              <span className="text-gray-700 font-medium">{skill}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default SkillsSection; 