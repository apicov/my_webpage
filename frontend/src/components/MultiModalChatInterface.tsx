import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage } from '../types';
import { chatWithAI } from '../services/api';

// Display content types
type DisplayType = 'welcome' | 'video' | 'image' | 'chart' | 'code' | 'markdown' | 'loading' | 'error';

interface DisplayContent {
  type: DisplayType;
  data?: any;
  metadata?: {
    title?: string;
    description?: string;
    source?: string;
  };
}

interface MultiModalChatInterfaceProps {
  onAgentSwitch?: (agentId: string) => void;
  userInfo?: any;
}

const MultiModalChatInterface: React.FC<MultiModalChatInterfaceProps> = ({ onAgentSwitch, userInfo }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [displayContent, setDisplayContent] = useState<DisplayContent>({
    type: 'welcome'
  });
  const [isDisplayExpanded, setIsDisplayExpanded] = useState(true);
  const [viewportHeight, setViewportHeight] = useState(typeof window !== 'undefined' ? window.innerHeight : 0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const isProcessingRef = useRef(false);

  const initialMessage = `Hi! I'm ${userInfo?.name || 'Your Name'}'s AI assistant. I'm here to provide information about his professional background, skills, and experience. I can help you learn more about his career, projects, and achievements. I can also show you live hardware demos, display sensor data, and more. What would you like to know?`;

  // Initialize with welcome message
  useEffect(() => {
    setMessages([{
      role: 'assistant',
      content: initialMessage
    }]);
  }, [userInfo?.name]);

  // Auto-scroll to bottom of messages - same as old ChatInterface
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Auto-scroll to bottom when messages change - clean and simple
  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);

  // Detect viewport height changes (mobile keyboard)
  useEffect(() => {
    const updateHeight = () => {
      setViewportHeight(window.innerHeight);
    };
    
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  // Check if mobile keyboard is likely open
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 640;
  const initialHeight = typeof window !== 'undefined' ? window.screen.height : 0;
  const isKeyboardOpen = isMobile && viewportHeight < initialHeight * 0.75;


  // Display Area Component
  const DisplayArea = () => {
    const renderContent = () => {
      switch (displayContent.type) {
        case 'welcome':
          return (
            <div className="h-full flex flex-col items-center justify-center text-white p-4">
              <div className="text-center max-w-lg">
                <div className="mb-4">
                  <div className="text-4xl mb-2">🤖</div>
                  <h3 className="text-xl font-semibold mb-2">AI Assistant Ready</h3>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs sm:text-sm">
                  <div className="bg-white/10 rounded p-2">🎥 Hardware Demos</div>
                  <div className="bg-white/10 rounded p-2">📊 Live Sensors</div>
                  <div className="bg-white/10 rounded p-2">💡 Project Info</div>
                  <div className="bg-white/10 rounded p-2">🧠 Technical Q&A</div>
                </div>
              </div>
            </div>
          );
          
        case 'video':
          return (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-contain"
              src={displayContent.data}
            />
          );
          
        case 'image':
          return (
            <div className="h-full flex items-center justify-center p-4">
              <img 
                src={displayContent.data} 
                alt={displayContent.metadata?.title || 'Display'} 
                className="max-w-full max-h-full object-contain rounded"
              />
            </div>
          );
          
        case 'code':
          return (
            <div className="h-full overflow-auto p-4">
              <pre className="text-green-400 font-mono text-xs sm:text-sm">
                <code>{displayContent.data}</code>
              </pre>
            </div>
          );
          
        case 'markdown':
          return (
            <div className="h-full overflow-auto p-4 text-white">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {displayContent.data}
              </ReactMarkdown>
            </div>
          );
          
        case 'chart':
          return (
            <div className="h-full flex items-center justify-center p-4">
              <div className="text-white text-center">
                <div className="text-3xl mb-2">📊</div>
                <p>Chart visualization will appear here</p>
                {displayContent.data && (
                  <div className="mt-4 text-sm">{JSON.stringify(displayContent.data)}</div>
                )}
              </div>
            </div>
          );
          
        case 'loading':
          return (
            <div className="h-full flex items-center justify-center">
              <div className="text-white text-center">
                <div className="loading-spinner mb-2"></div>
                <p className="text-sm">{displayContent.metadata?.description || 'Loading...'}</p>
              </div>
            </div>
          );
          
        case 'error':
          return (
            <div className="h-full flex items-center justify-center p-4">
              <div className="text-red-400 text-center">
                <div className="text-3xl mb-2">⚠️</div>
                <p>{displayContent.data || 'An error occurred'}</p>
              </div>
            </div>
          );
          
        default:
          return null;
      }
    };

    return (
      <div className={`
        bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 
        transition-all duration-300 relative overflow-hidden flex-shrink-0
        ${isDisplayExpanded 
          ? 'h-48 sm:h-56 md:h-64 lg:h-72' 
          : 'h-10'
        }
      `}>
        {isDisplayExpanded ? (
          <>
            {renderContent()}
            {displayContent.metadata?.title && (
              <div className="absolute top-2 left-2 bg-black/50 text-white px-2 py-1 rounded text-sm">
                {displayContent.metadata.title}
              </div>
            )}
            <button
              onClick={() => setIsDisplayExpanded(false)}
              className="absolute top-2 right-2 p-1 bg-black/50 hover:bg-black/70 text-white rounded transition-colors"
              title="Minimize display"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
              </svg>
            </button>
          </>
        ) : (
          <div className="h-full flex items-center justify-between px-3 bg-gray-800">
            <span className="text-white text-sm">Display minimized</span>
            <button
              onClick={() => setIsDisplayExpanded(true)}
              className="p-1 hover:bg-white/20 text-white rounded transition-colors"
              title="Expand display"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            </button>
          </div>
        )}
      </div>
    );
  };

  // Process message and detect intent
  const processMessage = async (message: string) => {
    // Simple intent detection - this will be replaced with agent orchestration
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('hardware') || lowerMessage.includes('led') || lowerMessage.includes('servo')) {
      setDisplayContent({
        type: 'loading',
        metadata: { description: 'Connecting to hardware...' }
      });
      // Simulate hardware connection
      setTimeout(() => {
        setDisplayContent({
          type: 'video',
          data: null, // Will be WebRTC stream
          metadata: { title: 'Hardware Control Stream' }
        });
      }, 1500);
    } else if (lowerMessage.includes('sensor') || lowerMessage.includes('temperature')) {
      setDisplayContent({
        type: 'chart',
        data: { temperature: 24.5, humidity: 45, co2: 420 },
        metadata: { title: 'Live Sensor Data' }
      });
    } else if (lowerMessage.includes('code') || lowerMessage.includes('example')) {
      setDisplayContent({
        type: 'code',
        data: `// Example code from your projects
function controlHardware(device, action) {
  const mqtt = new MQTTClient();
  mqtt.connect('mqtt://broker.local');
  
  mqtt.publish(\`devices/\${device}/control\`, {
    action: action,
    timestamp: Date.now()
  });
  
  return mqtt.subscribe(\`devices/\${device}/status\`);
}`,
        metadata: { title: 'Code Example' }
      });
    } else if (lowerMessage.includes('project') || lowerMessage.includes('portfolio')) {
      setDisplayContent({
        type: 'markdown',
        data: `## My Projects\n\n### 🤖 Hardware Control System\nReal-time control of IoT devices using WebRTC and MQTT\n\n### 📊 Environmental Monitor\nTrack temperature, humidity, and CO2 levels\n\n### 🧠 AI Research\nMachine learning applications in robotics`,
        metadata: { title: 'Project Overview' }
      });
    }
  };

  // Clear chat function
  const clearChat = () => {
    setMessages([{
      role: 'assistant',
      content: initialMessage
    }]);
    setDisplayContent({ type: 'welcome' });
  };

  // Send message
  const sendMessage = async () => {
    if (!inputMessage.trim() || isSending || isProcessingRef.current) return;

    // Prevent duplicate calls
    const messageToSend = inputMessage.trim();
    isProcessingRef.current = true;
    setInputMessage('');
    setIsSending(true);

    const userMessage: ChatMessage = {
      role: 'user',
      content: messageToSend
    };

    // Add user message first
    setMessages(prev => [...prev, userMessage]);

    try {
      // Process display content based on message
      await processMessage(messageToSend);
      
      // Get AI response
      const response = await chatWithAI([...messages, userMessage]);
      
      if (response && response.response) {
        let assistantMessages: any[] = [];
        
        if (Array.isArray(response.response)) {
          assistantMessages = response.response;
        } else if (response.response) {
          assistantMessages = [{ content: response.response }];
        }
        
        // Take the last message from response array
        if (assistantMessages.length > 0) {
          const lastMessage = assistantMessages[assistantMessages.length - 1];
          
          if (lastMessage && lastMessage.content) {
            const assistantMessage: ChatMessage = {
              role: 'assistant',
              content: lastMessage.content,
              media: (response as any).media
            };
            setMessages(prev => [...prev, assistantMessage]);
          }
        }
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.'
      }]);
    } finally {
      setIsSending(false);
      isProcessingRef.current = false;
    }
  };

  return (
    <div 
      className="bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200 w-full lg:max-w-4xl lg:mx-auto"
      style={isKeyboardOpen ? {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: `${viewportHeight}px`,
        display: 'flex',
        flexDirection: 'column',
        zIndex: 1000,
        borderRadius: 0
      } : {}}
    >

      {/* Display Area */}
      <DisplayArea />

      {/* Chat Messages - Using same structure as old ChatInterface */}
      <div 
        ref={chatMessagesRef}
        className="chat-container overflow-y-auto p-4 bg-gray-50 smooth-scroll"
        style={isKeyboardOpen ? { flex: 1, minHeight: 0 } : {}}
      >
        {messages.map((message, index) => (
          <div key={index} className="mb-4 message-fade-in">
            <div className={`flex items-start ${message.role === 'user' ? 'justify-end' : ''}`}>
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full gradient-bg flex items-center justify-center mr-3 mt-1">
                  <i className="fas fa-robot text-white text-sm"></i>
                </div>
              )}
              
              <div className={`p-3 rounded-lg shadow-sm max-w-md ${
                message.role === 'user' 
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white mr-3' 
                  : 'bg-white'
              }`}>
                <p 
                  className={`text-left whitespace-pre-line ${
                    message.role === 'user' ? '' : 'text-gray-800'
                  }`}
                >
                  {message.content}
                </p>
                {message.media && (
                  <div className="media-content">
                    {message.media.type === 'image' && (
                      <img 
                        src={message.media.url} 
                        alt={message.media.alt || 'Image'} 
                        className="media-image"
                      />
                    )}
                    {message.media.type === 'video' && (
                      <video className="media-video" controls>
                        <source src={message.media.url} type={message.media.mimeType || 'video/mp4'} />
                      </video>
                    )}
                  </div>
                )}
              </div>
              
              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-gray-400 flex items-center justify-center mt-1">
                  <i className="fas fa-user text-white text-sm"></i>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {/* Typing Indicator */}
        {isSending && (
          <div className={`mb-4 typing-indicator ${isSending ? 'show' : ''}`}>
            <div className="flex items-start">
              <div className="w-8 h-8 rounded-full gradient-bg flex items-center justify-center mr-3">
                <i className="fas fa-robot text-white text-sm"></i>
              </div>
              <div className="bg-white p-3 rounded-lg shadow-sm">
                <div className="flex space-x-1">
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Chat Input - Same structure as old */}
      <div 
        className="p-4 border-t border-gray-200 bg-white"
        style={isKeyboardOpen ? { flexShrink: 0 } : {}}
      >
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Ask about projects, demos, or technical topics..."
            disabled={isSending}
            className="flex-1 px-3 py-2 text-sm border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:bg-gray-100"
            style={{ fontSize: '16px' }} // Prevent zoom on iOS
          />
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isSending}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed text-sm font-medium"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default MultiModalChatInterface;