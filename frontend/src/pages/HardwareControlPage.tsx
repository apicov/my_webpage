import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ChatWithVideoInterface from '../components/ChatWithVideoInterface';

const HardwareControlPage: React.FC = () => {
  const navigate = useNavigate();
  const [hardwareStatus, setHardwareStatus] = useState<Record<string, any>>({});

  const handleHardwareStatusUpdate = (status: Record<string, any>) => {
    setHardwareStatus(status);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-4 lg:mb-6">
          <button
            onClick={() => navigate('/')}
            className="flex items-center text-blue-600 hover:text-blue-800 transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span className="text-sm lg:text-base">Back to Home</span>
          </button>
          
          <h1 className="text-xl lg:text-2xl font-bold text-gray-800 text-center">
            Hardware Control Center
          </h1>
          
          <div className="w-20"></div> {/* Spacer for balance */}
        </div>

        {/* Main Content - Single integrated interface */}
        <div className="h-[calc(100vh-120px)] max-w-4xl mx-auto">
          <ChatWithVideoInterface onHardwareStatusUpdate={handleHardwareStatusUpdate} />
        </div>

        {/* Optional: Hardware Status Sidebar for Desktop */}
        {Object.keys(hardwareStatus).length > 0 && (
          <div className="hidden lg:block fixed top-20 right-4 bg-white rounded-lg shadow-lg p-4 w-48">
            <h3 className="font-semibold text-gray-700 mb-2 text-sm">Hardware Status</h3>
            <div className="space-y-2 text-xs">
              {Object.entries(hardwareStatus).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-gray-600 capitalize">{key}:</span>
                  <span className="font-medium text-gray-800">{value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HardwareControlPage;