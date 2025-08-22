import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ProjectCard from './ProjectCard';
import { Project } from '../types';

interface ProjectsSectionProps {
  projects: Project[];
}

const ProjectsSection: React.FC<ProjectsSectionProps> = ({ projects }) => {
  const navigate = useNavigate();
  const [filter, setFilter] = useState<'all' | 'completed' | 'in-progress'>('all');
  
  const filteredProjects = projects.filter(project => {
    if (filter === 'all') return true;
    return project.status === filter;
  });
  
  const featuredProjects = filteredProjects.filter(p => p.featured);
  const regularProjects = filteredProjects.filter(p => !p.featured);
  const sortedProjects = [...featuredProjects, ...regularProjects];
  
  return (
    <section className="py-16 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-800 mb-4">Projects</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Explore my portfolio of projects showcasing various technologies and problem-solving approaches
          </p>
        </div>
        
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded-md transition-colors ${
                filter === 'all'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              All Projects
            </button>
            <button
              onClick={() => setFilter('completed')}
              className={`px-4 py-2 rounded-md transition-colors ${
                filter === 'completed'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              Completed
            </button>
            <button
              onClick={() => setFilter('in-progress')}
              className={`px-4 py-2 rounded-md transition-colors ${
                filter === 'in-progress'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              In Progress
            </button>
          </div>
        </div>
        
        {sortedProjects.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-gray-500 text-lg">No projects found in this category.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sortedProjects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onClick={() => navigate(`/project/${project.id}`)}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
};

export default ProjectsSection;