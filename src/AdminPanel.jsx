import { useCallback, useEffect, useRef, useState } from 'react'
import { adminApi, auth, previewMode } from './cloud'
import AnalysisDetail from './AnalysisDetail'
import './Dashboard.css'

const readAdminImage=(path,options)=>adminApi('/admin'+path,options)
export default function AdminPanel() {
  const accessEpoch=useRef(0)
  const [authorized,setAuthorized]=useState(false)
  const [session,setSession]=useState(null)
  const [notice,setNotice]=useState('')
  const [checking,setChecking]=useState(true)
  const [localAvailable,setLocalAvailable]=useState(false)
  const [rows,setRows]=useState([])
  const [selected,setSelected]=useState(null)
  const [error,setError]=useState('')
  const [busy,setBusy]=useState(false)
  const [search,setSearch]=useState('')
  const [limit,setLimit]=useState(50)
  const clearAccess=useCallback(err=>{setError(err.message);if([401,403].includes(err.status)){setAuthorized(false);setRows([]);setSelected(null)}},[])
  const load=useCallback(async()=>{
    const epoch=accessEpoch.current
    const batches=await Promise.all(Array.from({length:limit/50},(_,page)=>adminApi(`/admin/analyses?limit=50&offset=${page*50}`)))
    if(epoch!==accessEpoch.current)return
    setRows([...new Map(batches.flat().map(row=>[row.id,row])).values()])
    setAuthorized(true)
  },[limit])
  useEffect(()=>{
    if(!auth)return
    const {data}=auth.auth.onAuthStateChange((_event,value)=>{accessEpoch.current++;setSession(value)})
    auth.auth.getSession().then(({data,error})=>{
      if(error)setError(error.message)
      setSession(data.session)
    })
    return()=>data.subscription.unsubscribe()
  },[])
  useEffect(()=>{
    let active=true
    const epochRef=accessEpoch
    epochRef.current++
    setAuthorized(false);setRows([]);setSelected(null);setChecking(true);setError('')
    async function setup(){
      try{
        const access=await adminApi('/admin/session')
        if(!active)return
        setLocalAvailable(access.local_login_enabled)
        if(access.authenticated || session || previewMode)await load()
      }catch(err){if(active)clearAccess(err)}
      finally{if(active)setChecking(false)}
    }
    setup();return()=>{active=false;epochRef.current++}
  },[load,session,clearAccess])
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
  async function emailLogin(event){
    event.preventDefault();setBusy(true);setError('');setNotice('')
    const email=new FormData(event.currentTarget).get('email')
    try{
      const {error}=await auth.auth.signInWithOtp({email,options:{shouldCreateUser:false,emailRedirectTo:window.location.origin}})
      if(error)throw error
      setNotice('Check your email. Open the sign-in link in this browser, then select Admin review. Only an authorized administrator account can view submitted scans.')
    }catch(err){setError(err.message)}finally{setBusy(false)}
  }
  async function passwordLogin(event){
    event.preventDefault();setBusy(true);setError('');setNotice('')
    const form=new FormData(event.currentTarget)
    try{
      const {error}=await auth.auth.signInWithPassword({email:form.get('username').trim(),password:form.get('password')})
      if(error)throw error
      event.target.reset()
    }catch(err){setError(err.message)}finally{setBusy(false)}
  }
  async function setPassword(event){
    event.preventDefault();setBusy(true);setError('');setNotice('')
    const form=event.currentTarget
    const values=new FormData(form)
    try{
      if(values.get('password')!==values.get('confirm'))throw new Error('Passwords do not match.')
      const {error}=await auth.auth.updateUser({password:values.get('password')})
      if(error)throw error
      form.reset();setNotice('Password saved. You can now sign in using your administrator email and password.')
    }catch(err){setError(err.message)}finally{setBusy(false)}
  }
  const current=rows.find(row=>row.id===selected)
  const filtered=rows.filter(row=>`${row.uploader?.email || ''} ${row.uploader?.display_name || ''} ${row.id} ${row.original_filename || ''} ${row.status}`.toLowerCase().includes(search.toLowerCase()))
  return <main className="research-shell">
    <header className="page-header"><span className="eyebrow">QXAI / ADMIN REVIEW</span><h1>The complete research picture<span>.</span></h1><p>Review submitted scans, model predictions, and the user who submitted each analysis.</p>
      <p className="research-note">Research predictions are not clinical diagnoses. The shared demo administrator login can view all submitted scans and uploader details.</p>
      {previewMode && <p className="preview-banner">Sample preview - illustrative users and results.</p>}
    </header>
    {error && <p role="alert" className="error-message">{error}<button onClick={()=>setError('')}>Dismiss</button></p>}
    {notice && <p role="status" className="dataset-resource">{notice}</p>}
    {checking ? <section className="panel" role="status">Checking administrator access…</section> : !authorized ? <section className="panel"><h2>Administrator sign-in</h2>
      {localAvailable ? <form onSubmit={login}>
        <p>Demo access uses the shared username and password. It opens all submitted scan records.</p>
        <label>Username<input name="username" autoComplete="username" required /></label>
        <label>Password<input name="password" type="password" autoComplete="current-password" required /></label>
        <button disabled={busy}>{busy?'Signing in…':'Sign in as admin'}</button>
      </form>:session ? <><p>Signed in as {session.user.email}. This account could not access administrator records. Use your approved administrator account, or retry if the service is starting.</p><button onClick={()=>load().catch(clearAccess)}>Retry access</button><button onClick={()=>auth.auth.signOut()}>Sign out and switch account</button></>:auth ? <div className="admin-auth-forms"><form onSubmit={passwordLogin}><p>Use your administrator email as your username.</p><label>Username (email)<input name="username" type="email" autoComplete="username" required /></label><label>Password<input name="password" type="password" autoComplete="current-password" required /></label><button disabled={busy}>{busy?'Signing in…':'Sign in as admin'}</button></form><details><summary>First time signing in or forgot your password?</summary><form onSubmit={emailLogin}><p>Request a sign-in link, open it in this browser, and select Admin review. Once signed in, use Set administrator password below your records.</p><label>Administrator email<input name="email" type="email" autoComplete="email" required /></label><button disabled={busy}>{busy?'Sending…':'Email sign-in link'}</button></form></details></div>:<p>Authentication is not configured. Please contact the site owner.</p>}
    </section>:<>
      <section className="panel admin-summary"><div><h2>All submitted analyses</h2><p>{rows.length} records loaded</p></div>
        <label>Search uploader or analysis<input type="search" value={search} onChange={event=>setSearch(event.target.value)} placeholder="Email, name, filename, or analysis ID" /></label>
        {session && <button onClick={()=>auth.auth.signOut()}>Sign out</button>}
        {localAvailable && <button onClick={async()=>{await adminApi('/admin/logout',{method:'POST'});setAuthorized(false);setRows([]);setSelected(null)}}>End admin session</button>}
      </section>
      {auth && !previewMode && <details className="panel admin-auth-forms"><summary>Set administrator password</summary><form onSubmit={setPassword}><p>Set a password for your signed-in account to use email and password sign-in next time.</p><label>New password<input name="password" type="password" autoComplete="new-password" minLength={12} required /></label><label>Confirm password<input name="confirm" type="password" autoComplete="new-password" minLength={12} required /></label><button disabled={busy}>{busy?'Saving…':'Save password'}</button></form></details>}
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
