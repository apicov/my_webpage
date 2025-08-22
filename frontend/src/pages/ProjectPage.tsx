import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Project } from '../types';
import { getProject } from '../services/api';

const ProjectPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchProject = async () => {
      if (!id) {
        navigate('/');
        return;
      }
      
      try {
        const projectData = await getProject(id);
        setProject(projectData);
      } catch (error) {
        console.error('Failed to fetch project:', error);
        navigate('/');
      } finally {
        setLoading(false);
      }
    };
    
    fetchProject();
  }, [id, navigate]);
  
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="loading-spinner"></div>
        <span className="ml-2">Loading project...</span>
      </div>
    );
  }
  
  if (!project) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-gray-500">Project not found</p>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <button
          onClick={() => navigate('/')}
          className="mb-6 flex items-center text-blue-600 hover:text-blue-800 transition-colors"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Home
        </button>
        
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          {project.thumbnail && (
            <div className="h-96 overflow-hidden">
              <img 
                src={project.thumbnail} 
                alt={project.title}
                className="w-full h-full object-cover"
              />
            </div>
          )}
          
          <div className="p-8">
            <div className="flex items-start justify-between mb-6">
              <div>
                <h1 className="text-4xl font-bold text-gray-800 mb-2">{project.title}</h1>
                <div className="flex items-center gap-4">
                  <span className={`text-sm px-3 py-1 rounded-full ${
                    project.status === 'completed' 
                      ? 'bg-green-100 text-green-800'
                      : project.status === 'in-progress'
                      ? 'bg-yellow-100 text-yellow-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {project.status === 'in-progress' ? 'In Progress' : 
                     project.status.charAt(0).toUpperCase() + project.status.slice(1)}
                  </span>
                  {project.featured && (
                    <span className="bg-yellow-100 text-yellow-800 text-sm px-3 py-1 rounded-full">
                      Featured Project
                    </span>
                  )}
                </div>
              </div>
              
              <div className="flex gap-4">
                {project.githubUrl && (
                  <a 
                    href={project.githubUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    View Code
                  </a>
                )}
                {project.liveUrl && (
                  <a 
                    href={project.liveUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                    View Live
                  </a>
                )}
                {project.demoUrl && (
                  <a 
                    href={project.demoUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Watch Demo
                  </a>
                )}
              </div>
            </div>
            
            {project.content ? (
              <div className="prose prose-lg max-w-none">
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  components={{
                    img: ({node, ...props}) => (
                      <img {...props} className="rounded-lg shadow-md my-4 w-full" />
                    ),
                    h2: ({node, ...props}) => (
                      <h2 {...props} className="text-2xl font-bold text-gray-800 mt-8 mb-4" />
                    ),
                    h3: ({node, ...props}) => (
                      <h3 {...props} className="text-xl font-semibold text-gray-800 mt-6 mb-3" />
                    ),
                    p: ({node, ...props}) => (
                      <p {...props} className="text-gray-600 leading-relaxed mb-4" />
                    ),
                    ul: ({node, ...props}) => (
                      <ul {...props} className="list-disc pl-6 mb-4 text-gray-600" />
                    ),
                    ol: ({node, ...props}) => (
                      <ol {...props} className="list-decimal pl-6 mb-4 text-gray-600" />
                    ),
                    code: ({node, ...props}: any) => {
                      const inline = !props.className?.includes('language-');
                      return inline ? 
                        <code {...props} className="bg-gray-100 px-1 py-0.5 rounded text-sm" /> :
                        <code {...props} className="block bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto" />
                    },
                    blockquote: ({node, ...props}) => (
                      <blockquote {...props} className="border-l-4 border-blue-500 pl-4 italic my-4" />
                    ),
                  }}
                >
                  {project.content}
                </ReactMarkdown>
              </div>
            ) : (
              <>
                {project.startDate && (
                  <div className="mb-6 text-gray-600">
                    <span className="font-semibold">Timeline:</span> {project.startDate}
                    {project.endDate && ` - ${project.endDate}`}
                  </div>
                )}
                
                <div className="mb-8">
                  <h2 className="text-2xl font-semibold text-gray-800 mb-4">Description</h2>
                  <p className="text-gray-600 leading-relaxed text-lg">
                    {project.longDescription || project.description}
                  </p>
                </div>
                
                <div>
                  <h2 className="text-2xl font-semibold text-gray-800 mb-4">Technologies Used</h2>
                  <div className="flex flex-wrap gap-3">
                    {project.technologies.map((tech, index) => (
                      <span 
                        key={index}
                        className="bg-blue-100 text-blue-800 px-4 py-2 rounded-lg font-medium"
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectPage;