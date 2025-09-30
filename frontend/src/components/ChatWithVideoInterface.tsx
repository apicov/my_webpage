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
  const [isDisconnecting, setIsDisconnecting] = useState(false);
  const [isVideoExpanded, setIsVideoExpanded] = useState(true);
  const [hardwareStatus, setHardwareStatus] = useState<Record<string, any>>({});
  const [userId] = useState(() => `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [gameState, setGameState] = useState<string>('waiting');
  const [gameBoard, setGameBoard] = useState<number[][]>([[0,0,0],[0,0,0],[0,0,0]]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [isSending, setIsSending] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState<number | null>(null);
  const [isInCooldown, setIsInCooldown] = useState(false);
  const [cooldownSeconds, setCooldownSeconds] = useState(0);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const frontendTimerRef = useRef<NodeJS.Timeout | null>(null);
  const cameraCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const cooldownTimerRef = useRef<NodeJS.Timeout | null>(null);
  const cameraViewingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const janusRef = useRef<any>(null);
  const streamingRef = useRef<any>(null);

  // Helper function to detect if device is desktop (has fine pointer and hover capability)
  const isDesktop = () => {
    return window.matchMedia('(pointer: fine) and (hover: hover)').matches;
  };

  // WebRTC configuration (kept for future use)
  // const webrtcConfig: WebRTCConfig = {
  //   iceServers: [
  //     { urls: 'stun:stun.l.google.com:19302' }
  //   ]
  // };

  // Check if camera is running and auto-connect if needed
  const checkCameraStatusAndConnect = async () => {
    try {
      const response = await fetch('/stream/status');
      const result = await response.json();

      if (result.status === 'success' && result.running && !isConnected && !isConnecting) {
        console.log('Camera detected running - auto-connecting using manual button flow...');

        // Stop periodic checking since we're about to connect
        if (cameraCheckIntervalRef.current) {
          clearInterval(cameraCheckIntervalRef.current);
          cameraCheckIntervalRef.current = null;
        }

        // Use the same flow as manual button - call initWebRTC
        await initWebRTC();
      }
    } catch (error) {
      console.error('Failed to check camera status:', error);
    }
  };

  // Connect to existing camera stream (WebRTC only, no camera start)
  const connectToExistingStream = async (janusUrl: string) => {
    setIsConnecting(true);
    try {
      await initializeJanus(janusUrl);
      startFrontendTimer();
    } catch (error) {
      console.error('Failed to connect to existing stream:', error);
      setIsConnected(false);
      setIsConnecting(false);
    }
  };

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

        // Stop auto-detection polling since we're now connected
        if (cameraCheckIntervalRef.current) {
          clearInterval(cameraCheckIntervalRef.current);
          cameraCheckIntervalRef.current = null;
        }

        // Initialize Janus WebRTC connection
        try {
          await initializeJanus(result.janus_url);
          // Start 5-minute frontend timer
          startFrontendTimer();
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

  // Handle endgame: 15-second camera viewing + 1-minute cooldown
  const handleEndgame = () => {
    console.log('Game ended - starting endgame sequence...');

    // Start 1-minute cooldown immediately
    startCooldown();

    // Keep camera on for 15 seconds to view final animation
    if (cameraViewingTimerRef.current) {
      clearTimeout(cameraViewingTimerRef.current);
    }

    cameraViewingTimerRef.current = setTimeout(() => {
      // Stop camera after 15 seconds
      disconnectWebRTC();
    }, 15000);
  };

  // Start 1-minute cooldown with countdown timer
  const startCooldown = () => {
    console.log('Starting 1-minute cooldown...');
    setIsInCooldown(true);
    setCooldownSeconds(60);

    if (cooldownTimerRef.current) {
      clearInterval(cooldownTimerRef.current);
    }

    cooldownTimerRef.current = setInterval(() => {
      setCooldownSeconds(prev => {
        if (prev <= 1) {
          // Cooldown finished
          setIsInCooldown(false);
          if (cooldownTimerRef.current) {
            clearInterval(cooldownTimerRef.current);
            cooldownTimerRef.current = null;
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  // Disconnect and stop camera stream
  const disconnectWebRTC = async () => {
    setIsDisconnecting(true);

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
      setIsDisconnecting(false);
      setTimeRemaining(null);

      // Stop all timers
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }

      if (frontendTimerRef.current) {
        clearTimeout(frontendTimerRef.current);
        frontendTimerRef.current = null;
      }

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }

    } catch (error) {
      console.error('Failed to stop camera stream:', error);
      setIsConnecting(false);
      setIsDisconnecting(false);
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

  // Check camera status periodically to detect auto-disconnect
  const startStatusPolling = () => {
    // Clear existing interval
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }

    // Check status every 30 seconds
    pingIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch('/api/stream/status');
        const result = await response.json();

        if (result.status === 'success') {
          if (!result.running && isConnected) {
            // Camera was auto-disconnected
            console.log('Camera auto-disconnected by server');
            handleAutoDisconnect();
          } else if (result.running) {
            // Update remaining time
            setTimeRemaining(result.time_remaining_minutes);
          }
        }
      } catch (error) {
        console.error('Failed to check camera status:', error);
      }
    }, 30000); // 30 seconds
  };

  // Start 5-minute frontend timer
  const startFrontendTimer = () => {
    // Clear existing timer
    if (frontendTimerRef.current) {
      clearTimeout(frontendTimerRef.current);
    }

    // Start countdown display
    let remainingSeconds = 5 * 60; // 5 minutes
    setTimeRemaining(Math.ceil(remainingSeconds / 60));

    const countdownInterval = setInterval(() => {
      remainingSeconds -= 1;
      setTimeRemaining(Math.ceil(remainingSeconds / 60));

      if (remainingSeconds <= 0) {
        clearInterval(countdownInterval);
      }
    }, 1000);

    // Set 5-minute auto-disconnect timer
    frontendTimerRef.current = setTimeout(() => {
      console.log('Frontend timer expired - disconnecting');
      clearInterval(countdownInterval);
      handleAutoDisconnect();
    }, 5 * 60 * 1000); // 5 minutes
  };

  // Handle automatic disconnection
  const handleAutoDisconnect = () => {
    console.log('Handling auto-disconnect');

    // Clean up connections
    if (streamingRef.current) {
      streamingRef.current.detach();
      streamingRef.current = null;
    }

    if (janusRef.current) {
      janusRef.current.destroy();
      janusRef.current = null;
    }

    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }

    // Update UI state
    setIsConnected(false);
    setIsConnecting(false);
    setTimeRemaining(null);

    // Clear video
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Stop all timers
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (frontendTimerRef.current) {
      clearTimeout(frontendTimerRef.current);
      frontendTimerRef.current = null;
    }
  };


  // Send message to TicTacToe game
  const sendMessage = async () => {
    if (!inputMessage.trim() || isSending || isInCooldown) return;

    // If this is the first message after endgame (cooldown just finished), reset the game
    if (gameState === 'endgame') {
      console.log('Resetting game after endgame...');
      setMessages([]);  // Clear old messages
      setGameState('welcome');
    }

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

        // Stop camera polling if game becomes blocked or has error
        if (response.state === 'busy' || response.state === 'error' || response.state === 'blocked') {
          if (cameraCheckIntervalRef.current) {
            clearInterval(cameraCheckIntervalRef.current);
            cameraCheckIntervalRef.current = null;
            console.log('Stopped camera polling - game is blocked or has error');
          }
        }

        // Handle endgame state
        if (response.state === 'endgame') {
          handleEndgame();
        }

        // Check if camera started after sending message (for tic-tac-toe auto-start)
        // Only start auto-detection if not already connected AND game is not blocked
        if (!isConnected && response.state !== 'busy' && response.state !== 'error' && response.state !== 'blocked') {
          // Only start camera detection if the game is actually progressing (not blocked)
          setTimeout(() => {
            checkCameraStatusAndConnect();

            // Start periodic checking if not already running
            if (!cameraCheckIntervalRef.current) {
              cameraCheckIntervalRef.current = setInterval(checkCameraStatusAndConnect, 3000);
            }
          }, 2000); // Wait 2 seconds for agent to start camera
        }
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.'
      }]);
    } finally {
      setIsSending(false);
      // Scroll to bottom and refocus the input field (desktop only)
      setTimeout(() => {
        if (messagesContainerRef.current) {
          const container = messagesContainerRef.current;
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
          });
        }
        // Only auto-focus on desktop devices
        if (isDesktop()) {
          inputRef.current?.focus();
        }
      }, 250);
    }
  };

  // Auto-scroll to bottom of messages
  useEffect(() => {
    const scrollToBottom = () => {
      if (messagesContainerRef.current) {
        // Use requestAnimationFrame to ensure DOM is fully updated
        requestAnimationFrame(() => {
          const container = messagesContainerRef.current;
          if (container) {
            container.scrollTo({
              top: container.scrollHeight,
              behavior: 'smooth'
            });
          }
        });
      }
    };

    // Multiple timing attempts to ensure scroll works
    const timer1 = setTimeout(scrollToBottom, 50);
    const timer2 = setTimeout(scrollToBottom, 200);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [messages]);

  // Auto-focus bottom and input on page load (desktop only)
  useEffect(() => {
    const timer = setTimeout(() => {
      if (messagesContainerRef.current) {
        const container = messagesContainerRef.current;
        container.scrollTo({
          top: container.scrollHeight,
          behavior: 'smooth'
        });
      }
      // Only auto-focus on desktop devices
      if (isDesktop()) {
        inputRef.current?.focus();
      }
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  // Additional auto-scroll when sending messages
  useEffect(() => {
    if (isSending) {
      const timer = setTimeout(() => {
        if (messagesContainerRef.current) {
          const container = messagesContainerRef.current;
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
          });
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isSending]);

  // Cleanup on unmount and page unload
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Try to reset game when user leaves
      if (userId) {
        navigator.sendBeacon('/api/tictactoe/reset',
          JSON.stringify({user_id: userId}));
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      // Clean up event listener
      window.removeEventListener('beforeunload', handleBeforeUnload);

      // Clean up all timers
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }

      if (frontendTimerRef.current) {
        clearTimeout(frontendTimerRef.current);
      }

      if (cameraCheckIntervalRef.current) {
        clearInterval(cameraCheckIntervalRef.current);
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
  }, [userId]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (cooldownTimerRef.current) {
        clearInterval(cooldownTimerRef.current);
      }
      if (cameraViewingTimerRef.current) {
        clearTimeout(cameraViewingTimerRef.current);
      }
    };
  }, []);

  return (
    <>
      {/* Custom styles for desktop responsive behavior */}
      <style>{`
        @media (min-width: 1024px) {
          [data-mobile-padding="true"] {
            padding-top: 1.5rem !important;
            padding-bottom: 1.5rem !important;
          }
          [data-lg-style] {
            top: auto !important;
          }
        }
      `}</style>

      <div className="bg-white rounded-lg shadow-lg flex flex-col overflow-hidden relative" style={{ height: '100vh', minHeight: '100vh' }}>
      {/* Header with Connection Status - Fixed at top on mobile, relative on desktop */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-3 flex justify-between items-center fixed top-0 left-0 right-0 z-10 lg:relative lg:top-auto lg:left-auto lg:right-auto lg:z-auto">
        <div>
          <h2 className="font-semibold">TicTacToe AI Game</h2>
          <div className="flex items-center gap-2 text-sm opacity-90">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <span>{
              isConnecting ? 'Connecting...' :
              isDisconnecting ? 'Disconnecting...' :
              isConnected ? 'Camera Live' : 'Camera Offline'
            }</span>
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

      {/* Video Section - Fixed position on mobile, relative on desktop */}
      <div className={`bg-black transition-all duration-300 fixed left-0 right-0 z-10 lg:relative lg:left-auto lg:right-auto lg:z-auto ${
        isVideoExpanded ? 'h-48 lg:h-64' : 'h-12'
      }`} style={{ top: '64px' }} data-lg-style="top: auto">{/* Note: style will be overridden on desktop */}
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
                    {/* Hidden for now - auto-detection handles connection
                    <button
                      onClick={initWebRTC}
                      className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
                    >
                      Connect Camera
                    </button>
                    */}
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
              <span>Camera {
                isConnecting ? 'Connecting...' :
                isDisconnecting ? 'Disconnecting...' :
                isConnected ? 'Live' : 'Offline'
              }</span>
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

      {/* Messages Area - Will naturally resize when keyboard appears */}
      <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0 lg:p-6" style={{
        paddingTop: isVideoExpanded ? '256px' : '112px', // Account for fixed header + video on mobile
        paddingBottom: '88px' // Account for fixed input section on mobile
      }} data-mobile-padding="true">{/* Custom CSS will handle desktop */}
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

        <div ref={messagesEndRef} className="h-1" />
      </div>

      {/* Input Section - Fixed at bottom on mobile, relative on desktop */}
      <div className="border-t bg-gray-50 p-4 fixed bottom-0 left-0 right-0 z-10 lg:relative lg:bottom-auto lg:left-auto lg:right-auto lg:z-auto">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder={isInCooldown
              ? `Please wait ${cooldownSeconds} seconds before starting a new game...`
              : "Type a message to start playing..."
            }
            disabled={isSending || isInCooldown}
            className="flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
          />
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isSending || isInCooldown}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
    </>
  );
};

export default ChatWithVideoInterface;