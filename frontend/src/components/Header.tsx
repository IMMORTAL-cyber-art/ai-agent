import { BookOpen } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function Header() {
  return (
    <header className="w-full py-6 px-8 max-w-7xl mx-auto flex items-center justify-between z-10">
      <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
        <div className="bg-indigo-600 p-2 rounded-lg text-white shadow-md">
          <BookOpen size={24} />
        </div>
        <h1 className="text-xl font-bold tracking-tight text-slate-800">
          Literature<span className="text-indigo-600">AI</span>
        </h1>
      </Link>
      <nav className="flex items-center gap-6 text-sm font-medium text-slate-600">
        <a href="https://github.com/your-repo" target="_blank" className="hover:text-indigo-600 transition-colors">Documentation</a>
      </nav>
    </header>
  )
}
