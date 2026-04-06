import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Results from './pages/Results'
import Header from './components/Header'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col relative overflow-hidden bg-slate-50">
        {/* Background decorative elements */ }
        <div className="absolute top-0 inset-x-0 h-64 bg-gradient-to-b from-indigo-100/50 to-transparent -z-10"></div>
        <Header />
        <main className="flex-grow flex flex-col">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
