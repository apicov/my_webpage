import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from '../types';
import { chatWithTicTacToe } from '../services/api';
// Temporarily commented out Janus import to test if it's breaking the site
// @ts-ignore - Janus library doesn't have proper TypeScript definitions
// const Janus = require('janus-gateway');

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
  const [userId] = useState(() => `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [gameState, setGameState] = useState<string>('waiting');
  const [gameBoard, setGameBoard] = useState<number[][]>([[0,0,0],[0,0,0],[0,0,0]]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isSending, setIsSending] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState<number | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const janusRef = useRef<any>(null);
  const streamingRef = useRef<any>(null);

  // WebRTC configuration (kept for future use)
  // const webrtcConfig: WebRTCConfig = {
  //   iceServers: [
  //     { urls: 'stun:stun.l.google.com:19302' }
  //   ]
  // };

  // Start camera stream via unified API
  const initWebRTC = async () => {
    setIsConnecting(true);
    try {
      // Call our Flask server to start the camera stream
      const response = await fetch('/api/stream/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const result = await response.json();

      if (result.status === 'success') {
        // Camera started successfully
        setIsConnected(true);
        setIsConnecting(false);

        // Initialize Janus WebRTC connection
        try {
          await initializeJanus(result.janus_url);
          // Start periodic ping to keep session alive (every 2 minutes)
          startPingInterval();
        } catch (janusError) {
          console.error('Janus connection failed:', janusError);
          setIsConnected(false);
          setIsConnecting(false);
        }

      } else {
        throw new Error(result.message || 'Failed to start camera stream');
      }

    } catch (error) {
      console.error('Failed to start camera stream:', error);
      setIsConnecting(false);
    }
  };

  // Disconnect and stop camera stream
  const disconnectWebRTC = async () => {
    setIsConnecting(true);

    try {
      // Call our Flask server to stop the camera stream
      const response = await fetch('/api/stream/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const result = await response.json();

      // Clean up Janus connections
      if (streamingRef.current) {
        streamingRef.current.detach();
        streamingRef.current = null;
      }

      if (janusRef.current) {
        janusRef.current.destroy();
        janusRef.current = null;
      }

      // Clean up WebRTC connection
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
        peerConnectionRef.current = null;
      }

      setIsConnected(false);
      setIsConnecting(false);
      setTimeRemaining(null);

      // Stop ping interval
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }

    } catch (error) {
      console.error('Failed to stop camera stream:', error);
      setIsConnecting(false);
    }
  };

  // Wait for Janus library to be available
  const waitForJanus = () => {
    return new Promise<void>((resolve, reject) => {
      let attempts = 0;
      const maxAttempts = 50; // 5 seconds with 100ms intervals - should be enough for local file

      const checkJanus = () => {
        attempts++;

        if ((window as any).Janus) {
          console.log(`Janus library loaded successfully after ${attempts} attempts`);
          resolve();
          return;
        }

        if (attempts >= maxAttempts) {
          console.error('Janus library not found after', maxAttempts, 'attempts');
          console.log('Available on window:', Object.keys(window).filter(k => k.toLowerCase().includes('janus')));
          reject(new Error(`Janus library failed to load within 5 seconds. Using local janus.js file.`));
          return;
        }

        setTimeout(checkJanus, 100);
      };

      // Check immediately
      checkJanus();
    });
  };

  // Initialize Janus WebRTC using window.Janus (loaded via CDN)
  const initializeJanus = async (janusUrl: string) => {
    // Wait for Janus to be available
    await waitForJanus();

    return new Promise<void>((resolve, reject) => {
      // Use global Janus from CDN script
      const Janus = (window as any).Janus;

      Janus.init({
        debug: false,
        callback: () => {
          const janus = new Janus({
            server: janusUrl,
            success: () => {
              console.log("Janus session created");
              janusRef.current = janus;

              // Attach to streaming plugin
              janus.attach({
                plugin: "janus.plugin.streaming",
                success: (pluginHandle: any) => {
                  console.log("Plugin attached");
                  streamingRef.current = pluginHandle;

                  // Start watching stream ID 1
                  pluginHandle.send({
                    message: { request: "watch", id: 1 }
                  });
                },
                error: (error: any) => {
                  console.error("Plugin attach error:", error);
                  reject(error);
                },
                onmessage: (_msg: any, jsep: any) => {
                  if (jsep) {
                    streamingRef.current.createAnswer({
                      jsep: jsep,
                      media: { audioSend: false, videoSend: false },
                      success: (jsep: any) => {
                        streamingRef.current.send({
                          message: { request: "start" },
                          jsep: jsep
                        });
                      },
                      error: (error: any) => {
                        console.error("WebRTC setup failed:", error);
                        reject(error);
                      }
                    });
                  }
                },
                onremotetrack: (track: MediaStreamTrack, _mid: any, on: boolean) => {
                  if (on && videoRef.current) {
                    console.log("Got remote track - displaying video");
                    const stream = new MediaStream([track]);
                    videoRef.current.srcObject = stream;
                    setIsConnected(true);
                    setIsConnecting(false);
                    resolve();
                  }
                }
              });
            },
            error: (error: any) => {
              console.error("Janus session error:", error);
              reject(error);
            }
          });
        }
      });
    });
  };

  // Start periodic ping to keep camera session alive
  const startPingInterval = () => {
    // Clear existing interval
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }

    // Ping every 2 minutes to keep session alive
    pingIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch('/api/stream/ping', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();
        if (result.status === 'success') {
          setTimeRemaining(result.time_remaining_minutes);
        }
      } catch (error) {
        console.error('Failed to ping camera session:', error);
      }
    }, 120000); // 2 minutes
  };


  // Send message to TicTacToe game
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
      // Send to TicTacToe endpoint
      const response = await chatWithTicTacToe(inputMessage, userId);

      if (response) {
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: response.message
        };
        setMessages(prev => [...prev, assistantMessage]);

        // Update game state
        setGameState(response.state);

        // Simple hardware status update
        setHardwareStatus({ game_state: response.state });
        onHardwareStatusUpdate?.({ game_state: response.state });
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

  // Auto-focus bottom on page load
  useEffect(() => {
    const timer = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 300); // Small delay to ensure component is fully rendered

    return () => clearTimeout(timer);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Clean up intervals
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }

      // Clean up Janus connections (but don't stop camera on unmount, let it auto-timeout)
      if (streamingRef.current) {
        streamingRef.current.detach();
      }

      if (janusRef.current) {
        janusRef.current.destroy();
      }

      // Clean up WebRTC
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
      }
    };
  }, []);

  return (
    <div className="bg-white rounded-lg shadow-lg flex flex-col overflow-hidden h-full" style={{ minHeight: '700px', maxHeight: '90vh' }}>
      {/* Header with Connection Status */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-3 flex justify-between items-center">
        <div>
          <h2 className="font-semibold">TicTacToe AI Game</h2>
          <div className="flex items-center gap-2 text-sm opacity-90">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <span>{isConnecting ? 'Connecting...' : isConnected ? 'Camera Live' : 'Camera Offline'}</span>
            <span className="mx-2">•</span>
            <span>Game: {gameState}</span>
            {isConnected && timeRemaining !== null && (
              <span className="text-xs opacity-75">({timeRemaining}m left)</span>
            )}
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
      <div className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            <svg className="w-12 h-12 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <p>Ready to play TicTacToe! 🎮</p>
            <p className="text-sm mt-2">Send any message to start a new game</p>
          </div>
        )}
        
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] md:max-w-[70%] lg:max-w-[60%] xl:max-w-[50%] px-4 py-3 rounded-lg ${
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
            placeholder="Type a message to start playing..."
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
        
      </div>
    </div>
  );
};

export default ChatWithVideoInterface;