import React from 'react';
import { Experience } from '../types';

interface ExperienceSectionProps {
  experience: Experience[];
}

const ExperienceSection: React.FC<ExperienceSectionProps> = ({ experience }) => {
  return (
    <section className="py-16 bg-gray-50">
      <div className="container mx-auto px-4">
        <h3 className="text-3xl font-bold text-center mb-12 text-gray-800">Professional Experience</h3>
        <div className="max-w-4xl mx-auto">
          {experience.map((exp, index) => (
            <div key={index} className="bg-white rounded-lg shadow-md p-6 mb-6 border-l-4 border-purple-500 hover:shadow-lg transition-shadow duration-300">
              <div className="flex flex-col md:flex-row md:justify-between md:items-start mb-4 space-y-2 md:space-y-0">
                <div className="flex-1 text-left">
                  <h4 className="text-xl font-semibold text-gray-800 mb-1 leading-tight text-left">{exp.role}</h4>
                  <p className="text-lg text-purple-600 font-medium mb-2 text-left">{exp.company}</p>
                </div>
                <div className="md:text-right">
                  <span className="inline-block bg-purple-100 text-purple-700 px-3 py-1 rounded-full text-sm font-medium">{exp.period}</span>
                </div>
              </div>
              <div className="border-t border-gray-100 pt-4">
                <p className="text-gray-600 leading-relaxed text-base text-left whitespace-pre-line">{exp.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ExperienceSection;