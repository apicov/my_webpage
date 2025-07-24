import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { chatWithAI } from '../services/api';

function ChatInterface({ userInfo }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showTypingIndicator, setShowTypingIndicator] = useState(false);
  const chatMessagesRef = useRef(null);
  const isProcessingRef = useRef(false);

  const initialMessage = `Hi! I'm ${userInfo?.name || 'Your Name'}'s AI assistant. I'm here to provide information about their professional background, skills, and experience. I can help you learn more about their career, projects, and achievements. What would you like to know?`;

  useEffect(() => {
    // Add initial message
    const assistantMessage = {
      role: 'assistant',
      content: initialMessage
    };
    setMessages([assistantMessage]);
  }, [userInfo?.name]);

  useEffect(() => {
    // Scroll to bottom when messages change
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);

  const formatMessageText = (text) => {
    // Escape HTML and convert \n to <br>
    const div = document.createElement('div');
    div.textContent = text;
    const escapedText = div.innerHTML;
    return escapedText.replace(/\n/g, '<br>');
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isTyping || isProcessingRef.current) return;

    // Prevent duplicate calls in StrictMode
    const messageToSend = inputMessage.trim();
    isProcessingRef.current = true;
    setInputMessage('');
    setIsTyping(true);
    setShowTypingIndicator(true);

    const userMessage = {
      role: 'user',
      content: messageToSend
    };

    // First, add the user message to the chat
    setMessages(prevMessages => {
      const allMessages = [...prevMessages, userMessage];
      return allMessages;
    });
    
    // Make the API call separately
    chatWithAI([...messages, userMessage]).then(response => {
      if (response && (response.status === 'success' || response.response)) {
        let assistantMessages = [];
        
        if (Array.isArray(response.response)) {
          assistantMessages = response.response;
        } else if (response.response) {
          assistantMessages = [response.response];
        } else {
          return;
        }
        
        // Only take the last message from the response array (like HTML version)
        if (assistantMessages.length === 0) {
          throw new Error('No assistant messages received');
        }
        
        const lastMessage = assistantMessages[assistantMessages.length - 1];
        
        if (!lastMessage || !lastMessage.content) {
          throw new Error('Invalid message format received');
        }
        
        const assistantMessage = {
          role: 'assistant',
          content: lastMessage.content,
          media: response.media
        };
        
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        console.log('Invalid response format:', response);
        throw new Error('Invalid response format');
      }
    }).catch(error => {
      console.error('Chat error:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.'
      };
      setMessages(prev => [...prev, errorMessage]);
    }).finally(() => {
      setIsTyping(false);
      setShowTypingIndicator(false);
      isProcessingRef.current = false;
    });
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !isTyping) {
      sendMessage();
    }
  };

  const clearChat = () => {
    const assistantMessage = {
      role: 'assistant',
      content: initialMessage
    };
    setMessages([assistantMessage]);
  };

  const renderMediaContent = (media) => {
    if (!media) return null;

    switch (media.type) {
      case 'image':
        return (
          <div className="media-content">
            <img 
              src={media.url} 
              alt={media.alt || 'Image'} 
              className="media-image"
            />
          </div>
        );
      case 'video':
        return (
          <div className="media-content">
            <video className="media-video" controls>
              <source src={media.url} type={media.mimeType || 'video/mp4'} />
              Your browser does not support the video tag.
            </video>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <section id="chat-section" className="w-full h-full flex items-center">
      <div className="w-full">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h3 className="text-3xl font-bold text-gray-800 mb-4">Ask My AI Assistant</h3>
            <p className="text-lg text-gray-600">
              Want to know more about my experience, skills, or projects? 
              Chat with my AI assistant
            </p>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
            {/* Chat Header */}
            <div className="gradient-bg text-white p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-full bg-white bg-opacity-20 flex items-center justify-center mr-3">
                    <i className="fas fa-robot text-white"></i>
                  </div>
                  <div className="text-left">
                    <h4 className="font-semibold text-left">Professional Assistant</h4>
                    <p className="text-sm opacity-90 text-left">Ask me anything about {userInfo?.name || 'Your Name'}'s background</p>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button 
                    onClick={clearChat} 
                    className="text-white hover:text-gray-200 transition-colors" 
                    title="Clear Chat"
                  >
                    <i className="fas fa-trash"></i>
                  </button>
                </div>
              </div>
            </div>

            {/* Chat Messages */}
            <div 
              ref={chatMessagesRef}
              className="chat-container overflow-y-auto p-4 bg-gray-50 smooth-scroll"
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
                        dangerouslySetInnerHTML={{ __html: formatMessageText(message.content) }}
                      />
                      {message.media && renderMediaContent(message.media)}
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
              {showTypingIndicator && (
                <div className="mb-4 typing-indicator show">
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

            {/* Chat Input */}
            <div className="p-4 border-t border-gray-200 bg-white">
              <div className="flex space-x-3">
                <input 
                  type="text" 
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about skills, experience, projects..." 
                  className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  disabled={isTyping}
                />
                <button 
                  onClick={sendMessage}
                  disabled={isTyping || !inputMessage.trim()}
                  className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-lg hover:from-blue-600 hover:to-purple-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <i className="fas fa-paper-plane"></i>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

ChatInterface.propTypes = {
  userInfo: PropTypes.shape({
    name: PropTypes.string,
    title: PropTypes.string,
    bio: PropTypes.string
  })
};

export default ChatInterface; 