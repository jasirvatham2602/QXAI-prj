import { useCallback, useEffect, useState } from 'react'
import { adminApi, auth, previewMode } from './cloud'
import AnalysisDetail from './AnalysisDetail'
import './Dashboard.css'

const readAdminImage=(path,options)=>adminApi('/admin'+path,options)
export default function AdminPanel() {
  const [authorized,setAuthorized]=useState(false)
  const [localAvailable,setLocalAvailable]=useState(false)
  const [rows,setRows]=useState([])
  const [selected,setSelected]=useState(null)
  const [error,setError]=useState('')
  const [busy,setBusy]=useState(false)
  const [search,setSearch]=useState('')
  const [limit,setLimit]=useState(50)
  const clearAccess=useCallback(err=>{setError(err.message);if([401,403].includes(err.status)){setAuthorized(false);setRows([]);setSelected(null)}},[])
  const load=useCallback(async()=>{
    const batches=await Promise.all(Array.from({length:limit/50},(_,page)=>adminApi(`/admin/analyses?limit=50&offset=${page*50}`)))
    setRows([...new Map(batches.flat().map(row=>[row.id,row])).values()])
    setAuthorized(true)
  },[limit])
  useEffect(()=>{
    let active=true
    async function setup(){
      try{
        const session=await adminApi('/admin/session')
        if(!active)return
        setLocalAvailable(session.local_login_enabled)
        const user=auth ? (await auth.auth.getSession()).data.session?.user : null
        if(session.authenticated || user?.app_metadata?.role==='qxai_admin' || previewMode)await load()
      }catch(err){if(active)setError(err.message)}
    }
    setup();return()=>{active=false}
  },[load])
  useEffect(()=>{
    if(!authorized)return
    const timer=setInterval(()=>load().catch(clearAccess),5000)
    return()=>clearInterval(timer)
  },[authorized,load,clearAccess])
  async function login(event){
    event.preventDefault();setBusy(true);setError('')
    const form=new FormData(event.currentTarget)
    try{
      await adminApi('/admin/login',{method:'POST',body:JSON.stringify({username:form.get('username'),password:form.get('password')})})
      await load()
    }catch(err){setError(err.message)}finally{setBusy(false)}
  }
  const current=rows.find(row=>row.id===selected)
  const filtered=rows.filter(row=>`${row.uploader?.email || ''} ${row.uploader?.display_name || ''} ${row.id} ${row.original_filename || ''} ${row.status}`.toLowerCase().includes(search.toLowerCase()))
  return <main className="research-shell">
    <header className="page-header"><span className="eyebrow">QXAI / ADMIN REVIEW</span><h1>The complete research picture<span>.</span></h1><p>Review submitted scans, model predictions, and the user who submitted each analysis.</p>
      <p className="research-note">Research predictions are not clinical diagnoses. Access is restricted to administrators.</p>
      {previewMode && <p className="preview-banner">Sample preview - illustrative users and results.</p>}
    </header>
    {error && <p role="alert" className="error-message">{error}<button onClick={()=>setError('')}>Dismiss</button></p>}
    {!authorized ? <section className="panel"><h2>Administrator sign-in</h2>
      {localAvailable ? <form onSubmit={login}>
        <p>Local administrator access on this computer.</p>
        <label>Username<input name="username" autoComplete="username" required /></label>
        <label>Password<input name="password" type="password" autoComplete="current-password" required /></label>
        <button disabled={busy}>{busy?'Signing in…':'Sign in as admin'}</button>
      </form>:<p>Sign in with a verified administrator account from the main page to access this panel.</p>}
    </section>:<>
      <section className="panel admin-summary"><div><h2>All submitted analyses</h2><p>{rows.length} records loaded</p></div>
        <label>Search uploader or analysis<input type="search" value={search} onChange={event=>setSearch(event.target.value)} placeholder="Email, name, filename, or analysis ID" /></label>
        {localAvailable && <button onClick={async()=>{await adminApi('/admin/logout',{method:'POST'});setAuthorized(false);setRows([]);setSelected(null)}}>End admin session</button>}
      </section>
      <div className="workspace-grid"><section className="panel"><h2>Scans and uploaders</h2>
        {!filtered.length && <p>No matching analyses.</p>}
        {filtered.map(row=><button className={'history-row '+(selected===row.id?'selected':'')} key={row.id} onClick={()=>setSelected(row.id)}>
          <strong>{row.uploader?.display_name || row.uploader?.email || 'Unknown uploader'}</strong>
          {row.uploader?.display_name && <span>{row.uploader.email}</span>}
          {row.original_filename && <span className="original-filename">{row.original_filename}</span>}
          <span>{new Date(row.created_at).toLocaleString()} · {row.status}</span>
          {row.result && <span>Classical: {row.result.classical.label} · Quantum: {row.result.quantum.label}</span>}
        </button>)}
        {rows.length===limit && <button onClick={()=>setLimit(old=>old+50)}>Load older analyses</button>}
      </section><section className="panel detail-panel" aria-live="polite">
        {current && <section className="uploader-card"><h2>Submitted by</h2><p><strong>{current.uploader?.display_name || 'Name not provided'}</strong></p><p>{current.uploader?.email || 'Email unavailable'}</p><small>User ID: {current.uploader?.id || current.owner_id}</small><small>Joined: {current.uploader?.joined_at ? new Date(current.uploader.joined_at).toLocaleString():'Not recorded'}</small></section>}
        <AnalysisDetail key={current?.id || 'none'} row={current} imageApi={readAdminImage} showDelete={false} />
      </section></div>
    </>}
  </main>
}
