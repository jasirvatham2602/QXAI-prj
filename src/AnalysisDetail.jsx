import { useEffect, useState } from 'react'
import { api, previewMode } from './cloud'
import { Icon } from './Icons'
import { modelsDisagree, reportHtml, reportDownloadName } from './dashboard-utils.mjs'

function download(content, type, name) {
  const url = URL.createObjectURL(new Blob([content], { type }))
  const link = document.createElement('a')
  link.href = url; link.download = name; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
function asDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}
function PrivateImage({ id, kind, label, imageApi = api }) {
  const [url, setUrl] = useState('')
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    let objectUrl
    imageApi(`/analyses/${id}/images/${kind}`, { signal: controller.signal }).then(blob => {
      objectUrl = URL.createObjectURL(blob)
      if (!controller.signal.aborted) setUrl(objectUrl)
    }).catch(error => { if (error.name !== 'AbortError') setFailed(true) })
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [id, kind, imageApi])
  return url ? <img className="analysis-image" src={url} alt={label} /> : <p>{failed ? 'Image could not be loaded.' : 'Loading image…'}</p>
}
export default function AnalysisDetail({ row, onDelete, imageApi = api, showDelete = true }) {
  const [reportBusy, setReportBusy] = useState(false)
  const [error, setError] = useState('')
  async function exportReport(format = 'pdf') {
    setReportBusy(true); setError('')
    try {
      const images = {}
      for (const kind of ['input','saliency']) {
        if (!previewMode || kind !== 'input') images[kind] = await asDataUrl(await imageApi(`/analyses/${row.id}/images/${kind}`))
      }
      if (format === 'pdf') {
        const { createPdfReport } = await import('./pdf-report.mjs')
        download(createPdfReport(row, images), 'application/pdf', reportDownloadName(row, 'pdf'))
      } else download(reportHtml(row, images), 'text/html', reportDownloadName(row, 'html'))
    } catch (err) { setError(err.message || 'Could not build the report') }
    finally { setReportBusy(false) }
  }
  if (!row) return <div className="empty-state"><span><Icon name="spark" size={36}/></span><h2>A closer look starts here</h2><p>Select an analysis to compare predictions<br/>and explore the model explanations.</p></div>
  return <>
    <div className="detail-heading"><div><span className="eyebrow">MODEL COMPARISON</span><h2>Analysis details</h2></div><span className={'status-badge '+row.status}>{row.status}</span></div>
    <p className="original-filename"><strong>Original filename:</strong> {row.original_filename || 'Not recorded for this older upload'}</p>
    {row.preview_note && <p className="research-note">{row.preview_note}</p>}
    <small>{new Date(row.created_at).toLocaleString()}{row.attempts > 0 ? ` · Attempt ${row.attempts} of 3` : ''}</small>
    {row.status === 'queued' && <p>{row.failure_category ? 'A temporary service failure occurred. A retry is scheduled.' : 'Waiting for an available worker.'}</p>}
    {row.status === 'processing' && <p>Processing both models. You can leave this page and return later.</p>}
    {row.status === 'failed' && <p>Processing could not finish. Please submit a new sample. Reason: {row.failure_category?.replaceAll('_',' ') || 'unknown'}</p>}
    {row.result && <>
      <p className="research-note">{modelsDisagree(row) ? 'The models disagree on this sample.' : 'The models agree on this sample.'}</p>
      <div className="model-grid">{['classical','quantum'].map(kind => <article className={'model-card '+kind} key={kind}>
        <h3>{kind === 'classical' ? 'Classical XAI' : 'Quantum XAI'}</h3><span className="model-prediction">{row.result[kind].label}</span><small>Predicted class</small>
        {row.result[kind].probabilities.map((value,i) => <div className="probability" key={i}><div><span>{['AD','CONTROL','PD'][i]}</span><strong>{(value*100).toFixed(1)}%</strong></div><div className="probability-track"><span style={{width:`${Math.max(0,Math.min(100,value*100))}%`}}/></div></div>)}
      </article>)}</div>
      <small>Scores are model outputs, not calibrated clinical confidence.</small>
    </>}
    <details className="image-disclosure" open={row.status !== 'complete'}><summary>Input research sample</summary><PrivateImage key={`${row.id}-input`} id={row.id} kind="input" label="Uploaded research sample" imageApi={imageApi} /></details>
    {row.status === 'complete' && <>
      <div className="section-heading"><h3>Saliency comparison</h3><span className="muted">Gradient attribution</span></div><PrivateImage key={`${row.id}-saliency`} id={row.id} kind="saliency" label="Classical and quantum saliency maps" imageApi={imageApi} />
      <p>Processing: {row.duration_seconds?.toFixed(2)} seconds · Model: {row.model_version}</p>
      {Object.keys(row.stage_timings || {}).length > 0 && <details><summary>Processing breakdown</summary><dl>{Object.entries(row.stage_timings).map(([key,value]) => <div key={key}><dt>{key.replaceAll('_',' ')}</dt><dd>{Number(value).toFixed(3)} seconds</dd></div>)}</dl></details>}
      <div className="report-actions"><button className="primary-button" disabled={reportBusy} onClick={() => exportReport('pdf')}><Icon name="download" size={16}/>{reportBusy ? 'Preparing report…' : 'Download PDF'}</button>
      <button disabled={reportBusy} onClick={() => exportReport('html')}>HTML report</button>
      <button onClick={() => download(JSON.stringify(row,null,2), 'application/json', reportDownloadName(row, 'json'))}>Export JSON</button></div>
      {error && <p role="alert">{error}</p>}
    </>}
    {showDelete && ['complete','failed'].includes(row.status) && <button disabled={previewMode} className="danger" onClick={() => onDelete(row.id)}>Delete analysis</button>}
  </>
}
