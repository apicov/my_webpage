// API Configuration for different environments
const getApiBaseUrl = () => {
  // Check if we have a custom API URL from environment
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  
  // Default behavior based on environment
  if (process.env.NODE_ENV === 'production') {
    // In production, API is served from same domain at /api
    return '';
  } else {
    // In development, Flask runs on port 5000
    return 'http://localhost:5000';
  }
};

const API_BASE = getApiBaseUrl();

// Helper function to handle API responses
const handleResponse = async (response) => {
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }
  return response.json();
};

// Helper function to handle network errors
const handleNetworkError = (error, operation) => {
  console.error(`${operation} error:`, error);
  
  if (error.name === 'TypeError' && error.message.includes('fetch')) {
    throw new Error(`Unable to connect to server. Please check if the backend is running.`);
  }
  
  throw error;
};

// Chat API
export const chatWithAI = async (messages) => {
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages: messages }),
      // Add timeout for better UX
      signal: AbortSignal.timeout(30000) // 30 second timeout
    });
    
    return handleResponse(response);
  } catch (error) {
    return handleNetworkError(error, 'Chat API');
  }
};

// Get user info
export const getUserInfo = async () => {
  try {
    const response = await fetch(`${API_BASE}/api/user-info`, {
      signal: AbortSignal.timeout(10000) // 10 second timeout
    });
    return handleResponse(response);
  } catch (error) {
    return handleNetworkError(error, 'Get user info');
  }
};

// IoT Device APIs (for future use)
export const getDevices = async () => {
  try {
    const response = await fetch(`${API_BASE}/api/devices`, {
      signal: AbortSignal.timeout(10000)
    });
    return handleResponse(response);
  } catch (error) {
    return handleNetworkError(error, 'Get devices');
  }
};

export const controlDevice = async (deviceId, command) => {
  try {
    const response = await fetch(`${API_BASE}/api/devices/${deviceId}/control`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ command }),
      signal: AbortSignal.timeout(15000)
    });
    return handleResponse(response);
  } catch (error) {
    return handleNetworkError(error, 'Control device');
  }
};

export const getDeviceData = async (deviceId) => {
  try {
    const response = await fetch(`${API_BASE}/api/devices/${deviceId}/data`, {
      signal: AbortSignal.timeout(10000)
    });
    return handleResponse(response);
  } catch (error) {
    return handleNetworkError(error, 'Get device data');
  }
};

// Debug function to check API configuration
export const getApiConfig = () => {
  return {
    environment: process.env.NODE_ENV,
    apiBaseUrl: API_BASE,
    customApiUrl: process.env.REACT_APP_API_URL
  };
}; 