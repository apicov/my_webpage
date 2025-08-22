import React from 'react';
import { useNavigate } from 'react-router-dom';
import ProjectCard from './ProjectCard';
import { Project } from '../types';

interface ProjectsSectionProps {
  projects: Project[];
}

const ProjectsSection: React.FC<ProjectsSectionProps> = ({ projects }) => {
  const navigate = useNavigate();
  
  // Sort projects: featured first, then by order
  const featuredProjects = projects.filter(p => p.featured);
  const regularProjects = projects.filter(p => !p.featured);
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
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sortedProjects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onClick={() => navigate(`/project/${project.id}`)}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default ProjectsSection;