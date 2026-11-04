import { useEffect, useState } from 'react'
import { auth, api, previewMode, checkReadiness, modelInfo } from './cloud'
import { historyQuery, visibleRows, modelsDisagree } from './dashboard-utils.mjs'
import AnalysisDetail from './AnalysisDetail'
import { Icon } from './Icons'
import './Dashboard.css'

const PAGE_SIZE = 50
const sampleUrl = new URL('./assets/synthetic-test.png', import.meta.url).href
export default function Home() {
  const [session, setSession] = useState(null)
  const [rows, setRows] = useState([])
  const [selected, setSelected] = useState(previewMode ? 'sample-complete' : null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [file, setFile] = useState(null)
  const [filter, setFilter] = useState('all')
  const [dates, setDates] = useState({ start: '', end: '' })
  const [pages, setPages] = useState(1)
  const [hasMore, setHasMore] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const [health, setHealth] = useState('checking')
  const [modelMeta,setModelMeta]=useState(null)
  const [dragging, setDragging] = useState(false)
  const [search, setSearch] = useState('')
  useEffect(() => {
    if (!auth) return
    auth.auth.getSession().then(({ data }) => setSession(data.session))
    const { data } = auth.auth.onAuthStateChange((event,value) => {
      setSession(value)
      if (event === 'SIGNED_OUT') { setRows([]); setSelected(null); setFile(null); setMetrics(null) }
    })
    return () => data.subscription.unsubscribe()
  }, [])
  useEffect(() => {
    let active = true
    const check = () => checkReadiness().then(ready => { if (active) setHealth(ready ? 'ready' : 'unavailable') }).catch(() => { if (active) setHealth('unavailable') })
    modelInfo().then(value=>{if(active)setModelMeta(value)})
    check(); const timer = setInterval(check, 30000)
    return () => { active = false; clearInterval(timer) }
  }, [])
  useEffect(() => {
    if (!session) return
    const controller = new AbortController()
    let polling = false
    async function refresh() {
      if (polling) return
      polling = true
      try {
        if (dates.start && dates.end && dates.start > dates.end) throw new Error('Start date must not be after the end date.')
        const batches = await Promise.all(Array.from({length:pages},(_,page) => api(historyQuery({...dates,page,limit:PAGE_SIZE}), {signal:controller.signal})))
        if (!controller.signal.aborted) {
          setRows([...new Map(batches.flat().map(row => [row.id,row])).values()])
          setHasMore(batches.at(-1).length === PAGE_SIZE)
        }
      } catch (err) { if (err.name !== 'AbortError') setError(err.message) }
      finally { polling = false }
    }
    refresh(); const timer = setInterval(refresh, 5000)
    return () => { controller.abort(); clearInterval(timer) }
  }, [session,dates,pages])
  async function signIn(event) {
    event.preventDefault(); setBusy(true); setError(''); setNotice('')
    try {
      const { error: authError } = await auth.auth.signInWithOtp({ email:new FormData(event.currentTarget).get('email'),options:{emailRedirectTo:window.location.origin} })
      if (authError) throw authError
      setNotice('Check your email for your sign-in link. Open it in the same browser as this app.')
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  async function chooseSample() {
    setError('')
    try {
      const response = await fetch(sampleUrl)
      if (!response.ok) throw new Error('Sample could not be loaded')
      setFile(new File([await response.blob()],'synthetic-pipeline-test.png',{type:'image/png'}))
    } catch (err) { setError(err.message) }
  }
  async function upload(event) {
    event.preventDefault(); setBusy(true); setError(''); setNotice('')
    const form = event.currentTarget
    try {
      if (!file?.size || file.size > 8*1024*1024) throw new Error('Choose a PNG or JPEG up to 8 MB.')
      const body = new FormData(); body.append('image',file)
      const row = await api('/analyses',{method:'POST',body})
      setRows(old => [row,...old]); setSelected(row.id); setFile(null); form.reset()
      setNotice('Analysis submitted. History updates automatically.')
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  async function remove(id) {
    if (!window.confirm('Permanently delete this analysis and its images?')) return
    try { await api(`/analyses/${id}`,{method:'DELETE'}); setRows(old => old.filter(row => row.id !== id)); setSelected(null) }
    catch (err) { setError(err.message) }
  }
  function changeDate(key,value) { setPages(1); setDates(old => ({...old,[key]:value})) }
  const current = rows.find(row => row.id === selected)
  return <main className="research-shell">
    <header className="page-header">
      {previewMode && <p className="preview-banner" role="status"><strong>Local sample preview</strong> · Illustrative data only. Sign-in, uploads, and deletion require Supabase. <a href="/">View actual app</a></p>}
      <div className="header-title"><div><span className="eyebrow">CLASSICAL + QUANTUM INTELLIGENCE</span><h1>Your research, in focus<span>.</span></h1>
      <p>Compare predictions. Explore explanations. Keep the evidence.</p></div><div className="header-orbit"><Icon name="spark" size={32}/><span>QXAI / LAB</span></div></div>
      <p className="research-note"><Icon name="info" size={16}/> Research workspace · Public or synthetic scans only. Predictions are not clinical diagnoses.</p>
      <p className="dataset-resource">Need a research sample? <a href="https://www.kaggle.com/datasets/jasirvatham2602/classifying-ad-and-pd-v3?select=3_cls_2" target="_blank" rel="noopener noreferrer">Download public MRI scans from Kaggle <Icon name="arrow" size={14}/></a><span>Choose a PNG or JPEG from the 3_cls_2 folder. These samples come from the project dataset and are not an independent evaluation.</span></p>
      <p role="status" className={`service-status ${health}`}>{health === 'ready' ? '● Model worker ready' : health === 'checking' ? 'Checking processing availability…' : 'Processing worker is starting or unavailable. Submitted analyses will wait in the queue.'}</p>
      {session && !previewMode && <button onClick={() => auth.auth.signOut()}>Sign out</button>}
    </header>
    {error && <p role="alert" className="error-message">{error}<button onClick={() => setError('')}>Dismiss</button></p>}
    {notice && <p role="status">{notice}</p>}
    {!auth ? <section className="panel"><h2>Workspace setup required</h2><p>Connect the authentication service and API using the deployment guide.</p></section> : !session ?
      <form className="panel sign-in-card" onSubmit={signIn}><h2>Sign in to your workspace</h2><label>Email address<input name="email" type="email" autoComplete="email" required /></label><button disabled={busy}>Email me a sign-in link</button></form> : <>
      <section className="summary-grid" aria-label="Loaded analysis summary">
        <div><span className="stat-icon"><Icon name="file"/></span><div><small>Analyses loaded</small><strong>{rows.length.toString().padStart(2,'0')}</strong></div><span className="stat-context">In this view</span></div>
        <div><span className="stat-icon mint"><Icon name="check"/></span><div><small>Completed</small><strong>{rows.filter(row=>row.status==='complete').length.toString().padStart(2,'0')}</strong></div><span className="stat-context">Ready to explore</span></div>
        <div><span className="stat-icon lilac"><Icon name="spark"/></span><div><small>Models compared</small><strong>02</strong></div><span className="stat-context">Classical + quantum</span></div>
      </section>
      <form className="panel upload-panel" onSubmit={upload}>
        <div className="upload-intro"><span className="eyebrow">START AN EXPLORATION</span><h2>New analysis</h2><p>One scan. Two perspectives.<br/>Compare model predictions and saliency maps.</p><button className="text-button" type="button" onClick={chooseSample}>Choose synthetic test image <Icon name="arrow" size={16}/></button><small>Synthetic samples test processing only.</small>{modelMeta && <small>Active bundle: {modelMeta.model_version}</small>}</div>
        <div className="upload-controls"><label className={'dropzone '+(dragging?'dragging':'')} onDragOver={event=>{event.preventDefault();setDragging(true)}} onDragLeave={()=>setDragging(false)} onDrop={event=>{event.preventDefault();setDragging(false);setFile(event.dataTransfer.files[0] || null)}}>
          <input name="image" type="file" accept="image/png,image/jpeg" required={!file} onChange={event=>setFile(event.target.files[0] || null)}/><span className="upload-icon"><Icon name={file?'file':'upload'} size={24}/></span><strong>{file?file.name:'Drop a research scan here'}</strong><span>{file?`${(file.size/1024).toFixed(0)} KB · Click to replace`:'or click to browse files'}</span><small>PNG or JPEG · up to 8 MB</small></label>
          <div className="upload-actions"><label className="consent"><input type="checkbox" required/> I am uploading a public or synthetic research sample.</label><button className="primary-button" disabled={busy || previewMode}>{busy?'Submitting…':'Run comparison'}<Icon name="arrow" size={17}/></button></div>
        </div>
      </form>
      {session.user.app_metadata?.role === 'qxai_admin' && <details className="panel operations-panel"><summary>Processing & service health</summary>
        <button onClick={async () => { try { setMetrics(await api('/operations')) } catch (err) { setError(err.message) } }}>Refresh aggregate metrics</button>
        {metrics && <>{metrics.ready_workers === 0 && <p role="status" className="error-message">No model worker is ready. Check the worker host.</p>}{metrics.oldest_queue_seconds > 300 && <p role="status">Queued work has been waiting longer than five minutes.</p>}
          <dl className="metrics-grid">{Object.entries(metrics).map(([key,value]) => <div key={key}><dt>{key.replaceAll('_',' ')}</dt><dd>{typeof value === 'number' ? Number(value.toFixed(2)) : '—'}</dd></div>)}</dl></>}
      </details>}
      <div className="section-heading"><div><span className="eyebrow">YOUR EVIDENCE LIBRARY</span><h2>Analysis workspace</h2></div><span className="muted">Histories update automatically</span></div><div className="workspace-grid"><section className="panel history-panel"><h2>Recent analyses</h2><label className="search-field">Find a scan<input type="search" value={search} onChange={event=>setSearch(event.target.value)} placeholder="Search filename…"/></label>
        <label>Show<select value={filter} onChange={event => setFilter(event.target.value)}>{['all','queued','processing','complete','failed','disagreement'].map(value => <option key={value}>{value}</option>)}</select></label>
        <div className="date-filters"><label>From<input type="date" value={dates.start} onChange={event => changeDate('start',event.target.value)} /></label><label>Through<input type="date" value={dates.end} min={dates.start || undefined} onChange={event => changeDate('end',event.target.value)} /></label></div>
        {(dates.start || dates.end) && <button onClick={() => {setDates({start:'',end:''});setPages(1)}}>Clear dates</button>}
        {!visibleRows(rows,filter).filter(row=>(row.original_filename || row.id).toLowerCase().includes(search.toLowerCase())).length && <p>No analyses match these filters.</p>}
        {visibleRows(rows,filter).filter(row=>(row.original_filename || row.id).toLowerCase().includes(search.toLowerCase())).map(row => <button className={'history-row '+(selected === row.id ? 'selected' : '')} key={row.id} onClick={() => setSelected(row.id)}>
          {row.original_filename && <span className="original-filename">{row.original_filename}</span>}
          <span>{new Date(row.created_at).toLocaleString()}</span><strong className={'status-badge '+row.status}>{row.status}{row.status === 'queued' && row.failure_category ? ' · retry scheduled' : ''}</strong>
          {row.result && <span>{row.result.classical.label} / {row.result.quantum.label}{modelsDisagree(row) ? ' · Models disagree' : ''}</span>}
        </button>)}
        {hasMore && <button onClick={() => setPages(old => old+1)}>Load older analyses</button>}
        <small>{rows.length} analyses loaded</small>
      </section><section className="panel detail-panel" aria-live="polite"><AnalysisDetail key={current?.id || 'none'} row={current} onDelete={remove} /></section></div>
    </>}
  </main>
}
