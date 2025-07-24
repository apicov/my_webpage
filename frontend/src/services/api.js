const API_BASE = process.env.REACT_APP_API_URL || '';

// Helper function to handle API responses
const handleResponse = async (response) => {
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};

// Chat API
export const chatWithAI = async (messages) => {
  try {
    console.log('Sending chat request to:', `${API_BASE}/api/chat`);
    console.log('Messages:', messages);
    console.log('Messages JSON:', JSON.stringify(messages, null, 2));
    
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages: messages })
    });
    
    console.log('Response status:', response.status);
    console.log('Response headers:', response.headers);
    
    return handleResponse(response);
  } catch (error) {
    console.error('Chat API error:', error);
    throw error;
  }
};

// Get user info
export const getUserInfo = async () => {
  try {
    const response = await fetch(`${API_BASE}/api/user-info`);
    return handleResponse(response);
  } catch (error) {
    console.error('Get user info error:', error);
    throw error;
  }
};

// IoT Device APIs (for future use)
export const getDevices = async () => {
  try {
    const response = await fetch(`${API_BASE}/api/devices`);
    return handleResponse(response);
  } catch (error) {
    console.error('Get devices error:', error);
    throw error;
  }
};

export const controlDevice = async (deviceId, command) => {
  try {
    const response = await fetch(`${API_BASE}/api/devices/${deviceId}/control`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ command })
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Control device error:', error);
    throw error;
  }
};

export const getDeviceData = async (deviceId) => {
  try {
    const response = await fetch(`${API_BASE}/api/devices/${deviceId}/data`);
    return handleResponse(response);
  } catch (error) {
    console.error('Get device data error:', error);
    throw error;
  }
}; 