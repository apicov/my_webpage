export interface UserInfo {
  name?: string;
  title?: string;
  bio?: string;
  skills?: string[];
  experience?: Experience[];
}

export interface Experience {
  role: string;
  company: string;
  period: string;
  description: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  media?: MediaContent;
}

export interface MediaContent {
  type: 'image' | 'video';
  url: string;
  alt?: string;
  mimeType?: string;
}

export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  success: boolean;
}