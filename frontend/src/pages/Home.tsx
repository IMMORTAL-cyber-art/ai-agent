import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Sparkles, Globe } from 'lucide-react'

const LANGUAGES = [
  { code: 'English', label: '🇬🇧 English' },
  { code: 'Hindi', label: '🇮🇳 Hindi (हिंदी)' },
  { code: 'Kannada', label: '🇮🇳 Kannada (ಕನ್ನಡ)' },
  { code: 'Tamil', label: '🇮🇳 Tamil (தமிழ்)' },
  { code: 'Telugu', label: '🇮🇳 Telugu (తెలుగు)' },
]

export default function Home() {
  const [topic, setTopic] = useState('')
  const [language, setLanguage] = useState('English')
  const navigate = useNavigate()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!topic.trim()) return
    navigate(`/results?topic=${encodeURIComponent(topic)}&language=${encodeURIComponent(language)}`)
  }

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 py-20 pb-32">
      <div className="max-w-3xl w-full text-center space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-1000">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-sm font-medium mb-4">
          <Sparkles size={16} />
          <span>AI-Powered Research Assistant</span>
        </div>
        
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900 leading-tight">
          Synthesize literature <br className="hidden md:block"/>
          in <span className="text-gradient">seconds</span>.
        </h1>
        
        <p className="text-lg md:text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Enter a research topic and let our AI fetch relevant papers, summarize them, and generate a comprehensive literature review instantly.
        </p>

        {/* Language Selector */}
        <div className="flex justify-center">
          <div className="flex items-center gap-2 bg-white border-2 border-slate-200 rounded-full px-5 py-2.5 shadow-sm hover:border-indigo-300 transition-colors">
            <Globe size={16} className="text-indigo-500" />
            <label htmlFor="language-select" className="text-sm font-semibold text-slate-600 whitespace-nowrap">Review Language:</label>
            <select
              id="language-select"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="bg-transparent text-sm font-bold text-indigo-700 outline-none cursor-pointer pr-2"
            >
              {LANGUAGES.map(l => (
                <option key={l.code} value={l.code}>{l.label}</option>
              ))}
            </select>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="mt-4 max-w-2xl mx-auto w-full relative group">
          <div className="absolute inset-y-0 left-0 pl-6 flex items-center pointer-events-none text-slate-400 group-focus-within:text-indigo-600 transition-colors">
            <Search size={22} />
          </div>
          <input
            type="text"
            className="block w-full pl-14 pr-40 py-5 text-lg bg-white border-2 border-slate-200 rounded-full shadow-sm hover:border-slate-300 focus:ring-4 focus:ring-indigo-100 focus:border-indigo-500 transition-all outline-none"
            placeholder="e.g. Impact of AI on Healthcare Diagnostics..."
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            required
          />
          <button
            type="submit"
            className="absolute inset-y-2 right-2 flex items-center gap-2 bg-slate-900 hover:bg-indigo-600 text-white px-6 py-2 rounded-full font-medium transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={!topic.trim()}
          >
            Generate Review
            <Sparkles size={18} />
          </button>
        </form>
        
        <div className="pt-4 text-sm text-slate-500 font-medium opacity-80">
          Powered by Semantic Scholar & Google Gemini
        </div>
      </div>
    </div>
  )
}
