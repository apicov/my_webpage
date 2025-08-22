import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from '../types';
import { chatWithAI } from '../services/api';

interface WebRTCConfig {
  iceServers: { urls: string }[];
}

interface ChatWithVideoInterfaceProps {
  onHardwareStatusUpdate?: (status: Record<string, any>) => void;
}

const ChatWithVideoInterface: React.FC<ChatWithVideoInterfaceProps> = ({ 
  onHardwareStatusUpdate 
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isVideoExpanded, setIsVideoExpanded] = useState(true);
  const [hardwareStatus, setHardwareStatus] = useState<Record<string, any>>({});
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isSending, setIsSending] = useState(false);

  // WebRTC configuration
  const webrtcConfig: WebRTCConfig = {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' }
    ]
  };

  // Initialize WebRTC connection
  const initWebRTC = async () => {
    setIsConnecting(true);
    try {
      const pc = new RTCPeerConnection(webrtcConfig);
      peerConnectionRef.current = pc;

      // Handle incoming tracks
      pc.ontrack = (event) => {
        if (videoRef.current && event.streams[0]) {
          videoRef.current.srcObject = event.streams[0];
        }
      };

      // Handle connection state changes
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'connected') {
          setIsConnected(true);
          setIsConnecting(false);
        } else if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
          setIsConnected(false);
          setIsConnecting(false);
        }
      };

      // For demo purposes, simulate connection after 2 seconds
      setTimeout(() => {
        setIsConnected(true);
        setIsConnecting(false);
        // Add a welcome message
        setMessages([{
          role: 'assistant',
          content: '🎥 Camera connected! I can help you control the connected hardware. Try asking me to turn on an LED or read sensor values.'
        }]);
      }, 2000);

    } catch (error) {
      console.error('Failed to initialize WebRTC:', error);
      setIsConnecting(false);
    }
  };

  // Disconnect WebRTC
  const disconnectWebRTC = () => {
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }
    setIsConnected(false);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  // Send message to hardware control agent
  const sendMessage = async () => {
    if (!inputMessage.trim() || isSending) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsSending(true);

    try {
      // Send to hardware control endpoint
      const response = await chatWithAI([...messages, userMessage]);
      
      if (response && response.response) {
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: response.response
        };
        setMessages(prev => [...prev, assistantMessage]);
        
        // Simulate hardware status update
        const updatedStatus = { ...hardwareStatus };
        if (inputMessage.toLowerCase().includes('led')) {
          updatedStatus.led = 'on';
        } else if (inputMessage.toLowerCase().includes('servo')) {
          updatedStatus.servo = '90°';
        } else if (inputMessage.toLowerCase().includes('sensor')) {
          updatedStatus.temperature = '24.5°C';
          updatedStatus.humidity = '45%';
        }
        
        setHardwareStatus(updatedStatus);
        onHardwareStatusUpdate?.(updatedStatus);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.'
      }]);
    } finally {
      setIsSending(false);
    }
  };

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnectWebRTC();
    };
  }, []);

  return (
    <div className="bg-white rounded-lg shadow-lg flex flex-col overflow-hidden h-full max-h-screen">
      {/* Header with Connection Status */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-3 flex justify-between items-center">
        <div>
          <h2 className="font-semibold">Hardware Control Assistant</h2>
          <div className="flex items-center gap-2 text-sm opacity-90">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <span>{isConnecting ? 'Connecting...' : isConnected ? 'Live' : 'Offline'}</span>
          </div>
        </div>
        
        {isVideoExpanded && (
          <button
            onClick={() => setIsVideoExpanded(false)}
            className="p-1 hover:bg-white/20 rounded transition-colors"
            title="Minimize video"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          </button>
        )}
      </div>

      {/* Video Section */}
      <div className={`bg-black transition-all duration-300 ${
        isVideoExpanded ? 'h-48 lg:h-64' : 'h-12'
      }`}>
        {isVideoExpanded ? (
          <div className="h-full relative">
            {isConnected ? (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                {isConnecting ? (
                  <div className="text-white text-center">
                    <div className="loading-spinner mb-2"></div>
                    <p className="text-sm">Connecting to camera...</p>
                  </div>
                ) : (
                  <div className="text-center text-gray-400">
                    <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    <p className="text-sm mb-2">Camera not connected</p>
                    <button
                      onClick={initWebRTC}
                      className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
                    >
                      Connect Camera
                    </button>
                  </div>
                )}
              </div>
            )}
            
            {/* Video Controls Overlay */}
            <div className="absolute top-2 right-2 flex gap-2">
              {isConnected && (
                <button
                  onClick={disconnectWebRTC}
                  className="p-2 bg-black/50 hover:bg-black/70 text-white rounded transition-colors"
                  title="Disconnect camera"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
          </div>
        ) : (
          // Minimized video bar
          <div className="h-full flex items-center justify-between px-4 bg-gray-900">
            <div className="flex items-center gap-3 text-white text-sm">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span>Camera {isConnected ? 'Live' : 'Offline'}</span>
              {Object.keys(hardwareStatus).length > 0 && (
                <div className="flex gap-2 text-xs text-gray-300">
                  {Object.entries(hardwareStatus).slice(0, 2).map(([key, value]) => (
                    <span key={key}>{key}: {value}</span>
                  ))}
                </div>
              )}
            </div>
            <button
              onClick={() => setIsVideoExpanded(true)}
              className="p-1 hover:bg-white/20 text-white rounded transition-colors"
              title="Expand video"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            <svg className="w-12 h-12 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <p>Connect your camera and start controlling hardware</p>
            <p className="text-sm mt-2">Try: "Turn on the LED" or "Read temperature sensor"</p>
          </div>
        )}
        
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] lg:max-w-[80%] px-4 py-3 rounded-lg ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
            </div>
          </div>
        ))}
        
        {isSending && (
          <div className="flex justify-start">
            <div className="bg-gray-100 px-4 py-3 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Section */}
      <div className="border-t bg-gray-50 p-4">
        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask me to control hardware..."
            disabled={isSending}
            className="flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
          />
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isSending}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
        
        {/* Quick Commands */}
        <div className="flex flex-wrap gap-2">
          <span className="text-xs text-gray-500 mr-2">Quick:</span>
          {['Turn on LED', 'Read sensors', 'Servo 90°', 'Status check'].map((cmd) => (
            <button
              key={cmd}
              onClick={() => setInputMessage(
                cmd === 'Servo 90°' ? 'Rotate servo to 90 degrees' :
                cmd === 'Status check' ? 'Show me all hardware status' : cmd
              )}
              disabled={isSending}
              className="text-xs px-3 py-1 bg-white hover:bg-gray-100 border rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {cmd}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ChatWithVideoInterface;