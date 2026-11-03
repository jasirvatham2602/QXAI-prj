import { NavLink, Routes, Route } from 'react-router-dom'
import Home from './Home'
import About from './About'
import AdminPanel from './AdminPanel'
import {BrandMark,Icon} from './Icons'
import './Dashboard.css'
export default function MyNavBar(){
  return <div className="app-layout">
    <aside className="qxai-nav"><NavLink className="brand" to="/"><BrandMark/><span>QXAI<small>RESEARCH LAB</small></span></NavLink>
      <span className="nav-caption">WORKSPACE</span><nav aria-label="Main navigation">
        <NavLink end to="/"><Icon name="grid"/>My analyses</NavLink>
        <NavLink to="/admin"><Icon name="shield"/>Admin review</NavLink>
        <NavLink to="/about"><Icon name="info"/>About the models</NavLink>
      </nav>
      <div className="sidebar-note"><Icon name="spark"/><strong>Explore the evidence.</strong><p>Classical & quantum models.<br/>One research workspace.</p><span>RESEARCH USE ONLY</span></div>
      <div className="sidebar-footer"><span className="tiny-dot"/>Private by design</div>
    </aside>
    <div className="app-content"><div className="topbar"><span>Research workspace <span className="topbar-slash">/</span> <strong>QXAI</strong></span><span className="privacy-label"><Icon name="shield" size={15}/> Private scan storage</span></div>
      <Routes><Route path="/" element={<Home/>}/><Route path="/about" element={<About/>}/><Route path="/admin" element={<AdminPanel/>}/></Routes>
    </div>
  </div>
}
