import { useEffect, useState, useRef } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { 
  ArrowLeft, FileText, BookOpen, AlertCircle, Sparkles, 
  Target, BarChart3, Fingerprint, Lightbulb, Copy, 
  Download, RotateCcw, Clock, ShieldCheck, Zap, Send, MessageSquare, User, Bot, History, Plus
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { generateReview, askQuestion, ReviewResponse, ChatMessage } from '../services/api'

const HISTORY_KEY = 'lit_review_history'

interface HistoryItem {
  topic: string
  language: string
  timestamp: string
  review: ReviewResponse
}

export default function Results() {
  const [searchParams] = useSearchParams()
  const topic = searchParams.get('topic')
  const language = searchParams.get('language') || 'English'
  const navigate = useNavigate()

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<ReviewResponse | null>(null)
  const [copied, setCopied] = useState(false)

  // Expandable papers state
  const [expandedPaperIdx, setExpandedPaperIdx] = useState<number | null>(null)

  // Q&A Chat state
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const chatEndRef = useRef<HTMLDivElement>(null)

  // History sidebar
  const [showHistory, setShowHistory] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>([])

  useEffect(() => {
    const stored = localStorage.getItem(HISTORY_KEY)
    if (stored) setHistory(JSON.parse(stored))
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory, chatLoading])

  useEffect(() => {
    if (!topic) { navigate('/'); return }

    const fetchReview = async () => {
      setLoading(true)
      setError(null)
      try {
        const result = await generateReview(topic, language)
        setData(result)
        // Save to history
        const item: HistoryItem = {
          topic,
          language,
          timestamp: new Date().toISOString(),
          review: result
        }
        const stored = localStorage.getItem(HISTORY_KEY)
        const existing: HistoryItem[] = stored ? JSON.parse(stored) : []
        const updated = [item, ...existing.filter(h => h.topic !== topic || h.language !== language)].slice(0, 10)
        localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
        setHistory(updated)
      } catch (err: any) {
        console.error(err)
        setError(err?.response?.data?.detail || "An unexpected error occurred while generating the review.")
      } finally {
        setLoading(false)
      }
    }

    fetchReview()
  }, [topic, language, navigate])

  const handleSendQuestion = async () => {
    if (!question.trim() || chatLoading || !topic) return
    const userMsg: ChatMessage = { role: 'user', content: question }
    const updatedHistory = [...chatHistory, userMsg]
    setChatHistory(updatedHistory)
    setQuestion('')
    setChatLoading(true)
    try {
      const res = await askQuestion(topic, question, updatedHistory, language)
      setChatHistory(prev => [...prev, { role: 'ai', content: res.answer }])
    } catch {
      setChatHistory(prev => [...prev, { role: 'ai', content: 'Sorry, there was an error answering your question. Please try again.' }])
    } finally {
      setChatLoading(false)
    }
  }

  const copyToClipboard = () => {
    if (!data?.structured_review) return
    const sr = data.structured_review
    const text = `Literature Review: ${topic}\n\nIntroduction:\n${sr.introduction}\n\nKey Themes:\n${sr.key_themes.join('\n')}\n\nConclusion:\n${sr.conclusion}`
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleRegenerate = () => {
    setLoading(true); setError(null); setData(null); setChatHistory([])
    generateReview(topic!, language).then(setData).catch(e => setError(e?.response?.data?.detail || "Error")).finally(() => setLoading(false))
  }

  const loadFromHistory = (item: HistoryItem) => {
    navigate(`/results?topic=${encodeURIComponent(item.topic)}&language=${encodeURIComponent(item.language)}`)
    setShowHistory(false)
  }

  const review = data?.structured_review

  return (
    <div className="flex-1 w-full max-w-[1400px] mx-auto px-4 lg:px-8 py-8 pb-32 flex flex-col xl:flex-row gap-8">
      
      {/* History Sidebar Overlay */}
      {showHistory && (
        <div className="fixed inset-0 z-50 flex">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setShowHistory(false)}></div>
          <div className="relative z-10 w-96 max-w-full bg-white h-full shadow-2xl overflow-y-auto p-6 ml-auto animate-in slide-in-from-right duration-300">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2"><History size={20} className="text-indigo-600" /> Review History</h3>
              <button onClick={() => setShowHistory(false)} className="text-slate-400 hover:text-slate-700 text-2xl font-bold leading-none">×</button>
            </div>
            {history.length === 0 ? (
              <p className="text-slate-500 text-sm">No history yet. Generate a few reviews to see them here!</p>
            ) : (
              <div className="space-y-3">
                {history.map((item, idx) => (
                  <button key={idx} onClick={() => loadFromHistory(item)} className="w-full text-left p-4 rounded-xl border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50 transition-all group">
                    <p className="font-semibold text-slate-800 group-hover:text-indigo-700 text-sm line-clamp-2">{item.topic}</p>
                    <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-400">
                      <span className="bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-medium">{item.language}</span>
                      <span>{new Date(item.timestamp).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute:'2-digit' })}</span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 min-w-0">
        
        {/* Top Navigation Bar */}
        <div className="flex items-center justify-between mb-6 print:hidden">
          <button onClick={() => navigate('/')} className="flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-indigo-600 transition-colors">
            <ArrowLeft size={16} /> Back to Search
          </button>
          <button onClick={() => setShowHistory(true)} className="flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-indigo-600 transition-colors bg-white border border-slate-200 shadow-sm px-3 py-1.5 rounded-full">
            <History size={15} /> History
          </button>
        </div>

        {/* Header */}
        <div className="mb-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
          <div className="flex items-center gap-3 mb-2">
            <h2 className="text-3xl lg:text-4xl font-bold text-slate-900">Literature Review</h2>
            {!loading && <span className="bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded-full font-bold">AI Synthesized</span>}
          </div>
          <h3 className="text-xl text-indigo-600 font-medium">"{topic}"</h3>
          <p className="text-sm text-slate-400 mt-1">🌐 Language: <span className="font-semibold text-slate-600">{language}</span></p>
        </div>

        {/* Metrics badges */}
        {data && !loading && (
          <div className="flex flex-wrap gap-3 mb-8 print:hidden animate-in fade-in zoom-in duration-500">
            <div className="flex items-center gap-2 bg-white border border-slate-200 shadow-sm rounded-full px-4 py-2 text-sm font-medium text-slate-600">
              <ShieldCheck size={16} className="text-emerald-500"/> Confidence: {review?.confidence_level}
            </div>
            <div className="flex items-center gap-2 bg-white border border-slate-200 shadow-sm rounded-full px-4 py-2 text-sm font-medium text-slate-600">
              <Zap size={16} className="text-amber-500"/> Complexity: {review?.complexity_level}
            </div>
            {data.generation_time_seconds && (
              <div className="flex items-center gap-2 bg-white border border-slate-200 shadow-sm rounded-full px-4 py-2 text-sm font-medium text-slate-600">
                <Clock size={16} className="text-blue-500"/> Generated in {data.generation_time_seconds}s
              </div>
            )}
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-r-lg shadow-sm flex gap-4">
            <AlertCircle className="text-red-500 shrink-0 mt-1" />
            <div><h3 className="text-red-800 font-bold text-lg">Analysis Failed</h3><p className="text-red-700 mt-1">{error}</p></div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="bg-white/60 backdrop-blur-md rounded-2xl p-12 shadow-sm border border-slate-100 text-center">
            <div className="relative inline-flex mb-8">
              <div className="w-20 h-20 border-4 border-indigo-100 border-t-indigo-600 rounded-full animate-spin"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <Sparkles className="text-indigo-600 w-8 h-8 animate-pulse" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-slate-900">Analyzing Research Papers</h3>
            <p className="text-slate-500 mt-3 max-w-md mx-auto">Fetching papers, cross-referencing frameworks, and building your structured synthesis in <strong>{language}</strong>...</p>
          </div>
        )}

        {/* Loaded Review */}
        {data && review && !loading && !error && (
          <div className="space-y-8 print:text-black">
            
            {/* Action Bar */}
            <div className="flex gap-3 justify-end border-b border-slate-200 pb-4 print:hidden">
              <button onClick={copyToClipboard} className="flex gap-2 items-center bg-white border shadow-sm hover:bg-slate-50 text-slate-700 px-4 py-2 rounded-lg text-sm font-medium transition-all">
                <Copy size={16} /> {copied ? 'Copied!' : 'Copy'}
              </button>
              <button onClick={() => window.print()} className="flex gap-2 items-center bg-white border shadow-sm hover:bg-slate-50 text-slate-700 px-4 py-2 rounded-lg text-sm font-medium transition-all">
                <Download size={16} /> Save PDF
              </button>
              <button onClick={() => navigate('/')} className="flex gap-2 items-center bg-emerald-600 hover:bg-emerald-700 shadow-md text-white px-4 py-2 rounded-lg text-sm font-medium transition-all">
                <Plus size={16} /> New Review
              </button>
              <button onClick={handleRegenerate} className="flex gap-2 items-center bg-white border shadow-sm hover:bg-slate-50 text-slate-700 px-4 py-2 rounded-lg text-sm font-medium transition-all">
                <RotateCcw size={16} /> Regenerate
              </button>
            </div>

            {/* Introduction */}
            <section className="bg-white rounded-2xl p-6 md:p-8 shadow-sm border border-slate-200">
              <div className="flex items-center gap-3 mb-4 text-indigo-600">
                <FileText size={24} />
                <h3 className="text-2xl font-bold text-slate-900">Background & Introduction</h3>
              </div>
              <p className="text-slate-700 leading-relaxed text-lg">{review.introduction}</p>
            </section>

            {/* Key Themes */}
            <section className="bg-white rounded-2xl p-6 md:p-8 shadow-sm border border-slate-200">
              <div className="flex items-center gap-3 mb-6 text-emerald-600">
                <Target size={24} />
                <h3 className="text-2xl font-bold text-slate-900">Key Themes Identified</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {review.key_themes.map((theme, idx) => (
                  <div key={idx} className="flex items-start gap-4 p-4 rounded-xl bg-slate-50 border border-slate-100">
                    <span className="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold">{idx + 1}</span>
                    <p className="text-slate-700 pt-1">{theme}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Comparative Analysis */}
            <section className="bg-white rounded-2xl p-6 md:p-8 shadow-sm border border-slate-200">
              <div className="flex items-center gap-3 mb-6 text-blue-600">
                <BarChart3 size={24} />
                <h3 className="text-2xl font-bold text-slate-900">Comparative Analysis</h3>
              </div>
              <div className="prose prose-slate max-w-none w-full
                prose-table:w-full prose-table:border-collapse prose-table:rounded-lg prose-table:overflow-hidden 
                prose-th:bg-slate-100 prose-th:p-4 prose-th:text-left prose-th:border-b-2 prose-th:border-slate-200
                prose-td:p-4 prose-td:border-b prose-td:border-slate-100">
                <ReactMarkdown>{review.comparative_analysis}</ReactMarkdown>
              </div>
            </section>

            {/* Research Gaps */}
            <section className="bg-red-50/50 rounded-2xl p-6 md:p-8 shadow-sm border border-red-100">
              <div className="flex items-center gap-3 mb-6 text-red-600">
                <Fingerprint size={24} />
                <h3 className="text-2xl font-bold text-slate-900">Research Gaps</h3>
              </div>
              <ul className="space-y-3">
                {review.research_gaps.map((gap, idx) => (
                  <li key={idx} className="flex items-start gap-3">
                    <div className="mt-1.5 w-2 h-2 rounded-full bg-red-400 shrink-0"></div>
                    <span className="text-slate-700 leading-relaxed">{gap}</span>
                  </li>
                ))}
              </ul>
            </section>

            {/* Conclusion */}
            <section className="bg-slate-900 rounded-2xl p-6 md:p-8 shadow-lg text-white print:text-black print:bg-white print:border">
              <h3 className="text-2xl font-bold mb-4">Conclusion</h3>
              <p className="text-slate-300 leading-relaxed text-lg">{review.conclusion}</p>
            </section>

            {/* ───────────── Interactive Q&A Section ───────────── */}
            <section className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden print:hidden">
              <div className="p-5 bg-gradient-to-r from-indigo-600 to-violet-600 flex items-center gap-3">
                <MessageSquare size={22} className="text-white" />
                <div>
                  <h3 className="text-lg font-bold text-white">Ask Follow-Up Questions</h3>
                  <p className="text-indigo-200 text-xs mt-0.5">AI thinks beyond the research papers to give you expert answers in <strong className="text-white">{language}</strong></p>
                </div>
              </div>

              {/* Chat Messages */}
              <div className="p-5 space-y-4 min-h-[180px] max-h-[450px] overflow-y-auto bg-slate-50">
                {chatHistory.length === 0 && (
                  <div className="text-center text-slate-400 py-8">
                    <MessageSquare size={32} className="mx-auto mb-3 opacity-40" />
                    <p className="text-sm font-medium">No questions yet</p>
                    <p className="text-xs mt-1">Ask anything about <span className="font-semibold text-indigo-500">"{topic}"</span></p>
                  </div>
                )}
                {chatHistory.map((msg, idx) => (
                  <div key={idx} className={`flex items-start gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-indigo-600' : 'bg-gradient-to-br from-violet-500 to-indigo-600'}`}>
                      {msg.role === 'user' ? <User size={16} className="text-white" /> : <Bot size={16} className="text-white" />}
                    </div>
                    <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-tr-sm' : 'bg-white text-slate-700 border border-slate-200 rounded-tl-sm'}`}>
                      {msg.content}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-gradient-to-br from-violet-500 to-indigo-600">
                      <Bot size={16} className="text-white" />
                    </div>
                    <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                      <div className="flex gap-1.5 items-center">
                        <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:0ms]"></div>
                        <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:150ms]"></div>
                        <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:300ms]"></div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Input Box */}
              <div className="p-4 bg-white border-t border-slate-100 flex gap-3">
                <input
                  type="text"
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSendQuestion()}
                  placeholder={`Ask a question about "${topic}" in ${language}...`}
                  className="flex-1 bg-slate-50 rounded-xl px-4 py-3 text-sm text-slate-800 border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-400 transition-all"
                  disabled={chatLoading}
                />
                <button
                  onClick={handleSendQuestion}
                  disabled={!question.trim() || chatLoading}
                  className="flex items-center justify-center w-11 h-11 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl transition-colors shadow-md"
                >
                  <Send size={18} />
                </button>
              </div>
            </section>
          </div>
        )}
      </div>

      {/* Right Sidebar: Insights + Sources */}
      <div className="w-full xl:w-[400px] shrink-0 pt-16 xl:pt-28 flex flex-col gap-8">

        {/* Quick Insights Panel */}
        {data && review && !loading && !error && (
          <div className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-2xl border border-indigo-100 overflow-hidden shadow-sm print:hidden">
            <div className="p-6 bg-indigo-600 text-white relative overflow-hidden">
              <Sparkles className="absolute right-[-10px] top-[-10px] w-24 h-24 text-indigo-500 opacity-50" />
              <div className="flex items-center gap-2 mb-3 relative z-10">
                <Lightbulb size={20} className="text-yellow-300" />
                <h4 className="font-bold text-lg">AI Generated Idea</h4>
              </div>
              <p className="text-indigo-100 text-sm leading-relaxed relative z-10 font-medium">{review.ai_idea}</p>
            </div>
            <div className="p-6">
              <h4 className="font-bold text-slate-900 mb-4">⚡ Quick Takeaways</h4>
              <ul className="space-y-3">
                {review.key_takeaways.map((point, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-slate-600 font-medium">
                    <span className="shrink-0 text-indigo-500 mt-0.5">•</span>
                    {point}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Source Papers */}
        <div className="print:hidden">
          <div className="flex items-center gap-2 mb-4 text-slate-800 font-bold text-xl border-b pb-4">
            <BookOpen size={24} className="text-indigo-600" />
            <h3>Source Material</h3>
          </div>
          {loading ? (
            <div className="space-y-4">
              {[1,2,3].map(i => (
                <div key={i} className="rounded-xl p-4 h-24 animate-pulse bg-slate-100 shadow-sm border border-slate-200"></div>
              ))}
            </div>
          ) : data?.papers ? (
            <div className="space-y-4 overflow-y-auto max-h-[800px] pr-2 pb-8">
              {data.papers.map((paper, idx) => (
                <div key={idx} className="bg-white rounded-xl p-5 border border-slate-200 hover:border-indigo-300 transition-colors group shadow-sm hover:shadow-md">
                  <h4 className="font-bold text-slate-900 group-hover:text-indigo-700 transition-colors line-clamp-2 text-sm leading-snug mb-2">{paper.title}</h4>
                  <p className="text-xs text-slate-500 mb-3 truncate">
                    {paper.authors.length > 0 ? paper.authors.join(', ') : 'Unknown Authors'}
                    {paper.year && ` • ${paper.year}`}
                  </p>
                  {paper.abstract ? (
                    <div className="mt-2">
                      <button onClick={() => setExpandedPaperIdx(expandedPaperIdx === idx ? null : idx)} className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 flex items-center gap-1 transition-colors">
                        <FileText size={14} /> {expandedPaperIdx === idx ? 'Hide Abstract' : 'View Abstract'}
                      </button>
                      {expandedPaperIdx === idx && (
                        <div className="mt-3 text-xs text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-100">{paper.abstract}</div>
                      )}
                    </div>
                  ) : (
                    <span className="text-xs text-slate-400 italic">No abstract provided</span>
                  )}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
