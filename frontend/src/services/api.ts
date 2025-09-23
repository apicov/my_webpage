import { ChatMessage, UserInfo, Project } from '../types';

// API Configuration for different environments
const getApiBaseUrl = (): string => {
  // Check if we have a custom API URL from environment
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  
  // Default behavior based on environment
  if (import.meta.env.MODE === 'production') {
    // In production, API is served from same domain at /api
    return '/api';
  } else {
    // In development, Vite proxy handles /api routes
    return '/api';
  }
};

const API_BASE = getApiBaseUrl();

// Helper function to handle API responses
const handleResponse = async <T = any>(response: Response): Promise<T> => {
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }
  return response.json() as Promise<T>;
};

// Helper function to handle network errors
const handleNetworkError = (error: unknown, operation: string): never => {
  console.error(`${operation} error:`, error);
  
  if (error instanceof TypeError && error.message.includes('fetch')) {
    throw new Error(`Unable to connect to server. Please check if the backend is running.`);
  }
  
  if (error instanceof Error) {
    throw error;
  }
  
  throw new Error(`Unknown error in ${operation}`);
};

// Chat API
export const chatWithAI = async (messages: ChatMessage[]): Promise<{ response: string }> => {
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages }),
      // Add timeout for better UX
      signal: AbortSignal.timeout(30000) // 30 second timeout
    });

    return handleResponse<{ response: string }>(response);
  } catch (error) {
    return handleNetworkError(error, 'Chat API');
  }
};

// TicTacToe API - Simple unified interface
interface TicTacToeMessage {
  message: string;
  state?: 'start' | 'reset' | 'playing' | 'finished' | 'busy' | 'error';
  user_id: string;
}

interface TicTacToeResponse {
  message: string;
  state: 'playing' | 'finished' | 'busy' | 'error';
}

export const chatWithTicTacToe = async (message: string, userId: string, state?: string): Promise<TicTacToeResponse> => {
  try {
    const payload: TicTacToeMessage = { message, user_id: userId };
    if (state) payload.state = state as any;

    const response = await fetch(`${API_BASE}/tictactoe/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(30000)
    });

    return handleResponse<TicTacToeResponse>(response);
  } catch (error) {
    return handleNetworkError(error, 'TicTacToe Chat API');
  }
};

// Get user info
export const getUserInfo = async (): Promise<UserInfo> => {
  try {
    const response = await fetch(`${API_BASE}/user-info`, {
      signal: AbortSignal.timeout(10000) // 10 second timeout
    });
    return handleResponse<UserInfo>(response);
  } catch (error) {
    return handleNetworkError(error, 'Get user info');
  }
};

// IoT Device APIs (for future use)
interface Device {
  id: string;
  name: string;
  type: string;
  status: string;
}

interface DeviceCommand {
  command: string;
  [key: string]: any;
}

interface DeviceData {
  [key: string]: any;
}

export const getDevices = async (): Promise<Device[]> => {
  try {
    const response = await fetch(`${API_BASE}/devices`, {
      signal: AbortSignal.timeout(10000)
    });
    return handleResponse<Device[]>(response);
  } catch (error) {
    return handleNetworkError(error, 'Get devices');
  }
};

export const controlDevice = async (deviceId: string, command: DeviceCommand): Promise<{ success: boolean }> => {
  try {
    const response = await fetch(`${API_BASE}/devices/${deviceId}/control`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(command),
      signal: AbortSignal.timeout(15000)
    });
    return handleResponse<{ success: boolean }>(response);
  } catch (error) {
    return handleNetworkError(error, 'Control device');
  }
};

export const getDeviceData = async (deviceId: string): Promise<DeviceData> => {
  try {
    const response = await fetch(`${API_BASE}/devices/${deviceId}/data`, {
      signal: AbortSignal.timeout(10000)
    });
    return handleResponse<DeviceData>(response);
  } catch (error) {
    return handleNetworkError(error, 'Get device data');
  }
};

// Projects API
export const getProjects = async (): Promise<Project[]> => {
  try {
    const response = await fetch(`${API_BASE}/projects`, {
      signal: AbortSignal.timeout(10000)
    });
    return handleResponse<Project[]>(response);
  } catch (error) {
    return handleNetworkError(error, 'Get projects');
  }
};

export const getProject = async (id: string): Promise<Project> => {
  try {
    const response = await fetch(`${API_BASE}/projects/${id}`, {
      signal: AbortSignal.timeout(10000)
    });
    return handleResponse<Project>(response);
  } catch (error) {
    return handleNetworkError(error, 'Get project');
  }
};

// Debug function to check API configuration
export const getApiConfig = () => {
  return {
    environment: import.meta.env.MODE,
    apiBaseUrl: API_BASE,
    customApiUrl: import.meta.env.VITE_API_URL
  };
};