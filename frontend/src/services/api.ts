import axios from 'axios'

export interface Paper {
  title: string
  abstract: string | null
  authors: string[]
  year: number | null
}

export interface StructuredReview {
  introduction: string
  key_themes: string[]
  comparative_analysis: string
  research_gaps: string[]
  conclusion: string
  key_takeaways: string[]
  ai_idea: string
  confidence_level: string
  complexity_level: string
}

export interface ReviewResponse {
  structured_review: StructuredReview
  papers: Paper[]
  generation_time_seconds: number | null
}

export interface ChatMessage {
  role: 'user' | 'ai'
  content: string
}

export interface QAResponse {
  answer: string
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'


export const generateReview = async (topic: string, language: string = "English"): Promise<ReviewResponse> => {
  const response = await axios.post(`${API_BASE_URL}/generate-review`, { topic, language })
  return response.data
}

export const askQuestion = async (topic: string, question: string, chat_history: ChatMessage[], language: string = "English"): Promise<QAResponse> => {
  const response = await axios.post(`${API_BASE_URL}/qa`, { 
    topic, 
    question, 
    chat_history,
    language
  })
  return response.data
}
