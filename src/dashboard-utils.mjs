export function reportDownloadName(row, extension) {
  const stem = String(row.original_filename || 'analysis').replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9._-]+/g, '_').slice(0,100) || 'analysis'
  const id = String(row.id || 'record').replace(/[^a-zA-Z0-9_-]/g, '').slice(0,8)
  return `qxai-${stem}-${id}.${extension}`
}
export function historyQuery({ start = '', end = '', page = 0, limit = 50 } = {}) {
  const query = new URLSearchParams({ limit: String(limit), offset: String(page * limit) })
  if (start) query.set('start', new Date(`${start}T00:00:00`).toISOString())
  if (end) {
    const exclusive = new Date(`${end}T00:00:00`)
    exclusive.setDate(exclusive.getDate() + 1)
    query.set('end', exclusive.toISOString())
  }
  return `/analyses?${query}`
}
export const modelsDisagree = row => Boolean(row.result && row.result.classical.label !== row.result.quantum.label)
export function visibleRows(rows, filter) {
  return rows.filter(row => filter === 'all' || (filter === 'disagreement' ? modelsDisagree(row) : row.status === filter))
}
export const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char])
export function reportHtml(row, images = {}) {
  const text = escapeHtml
  const scoreRows = ['classical', 'quantum'].map(kind => {
    const model = row.result?.[kind]
    return model ? `<tr><td>${text(kind)}</td><td>${text(model.label)}</td>${model.probabilities.map(p => `<td>${text((p * 100).toFixed(1))}%</td>`).join('')}</tr>` : ''
  }).join('')
  const imageHtml = ['input', 'saliency'].map(kind => {
    // Only embed generated data URLs; never arbitrary URLs or markup from a record.
    const source = images[kind]
    return /^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=]+$/.test(source || '')
      ? `<h2>${kind === 'input' ? 'Input sample' : 'Saliency comparison'}</h2><img alt="${kind}" src="${source}"/>` : ''
  }).join('')
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"><title>QXAI analysis report</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:24px;color:#173c50}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #cbd7df;text-align:left}img{max-width:100%;max-height:700px;object-fit:contain}.note{padding:16px;background:#edf4f6}dt{font-weight:700}@media print{body{margin:0}img,table{break-inside:avoid}}</style></head><body><h1>QXAI research analysis</h1><p class="note">Research demonstration. Model predictions are not clinical diagnoses. Probability scores are not calibrated clinical confidence.</p>${row.preview_note ? `<p>${text(row.preview_note)}</p>` : ''}<dl><dt>Original filename</dt><dd>${text(row.original_filename || 'Not recorded for this older upload')}</dd><dt>Analysis</dt><dd>${text(row.id)}</dd><dt>Created</dt><dd>${text(row.created_at)}</dd><dt>Status</dt><dd>${text(row.status)}</dd><dt>Model version</dt><dd>${text(row.model_version || 'Not recorded')}</dd><dt>Preprocessing version</dt><dd>${text(row.preprocessing_version || 'Not recorded')}</dd><dt>Processing time</dt><dd>${text(row.duration_seconds?.toFixed(2) ?? 'Not recorded')} seconds</dd></dl><h2>Model comparison</h2><p>${modelsDisagree(row) ? 'The models disagree.' : 'The models agree.'}</p><table><thead><tr><th>Model</th><th>Prediction</th><th>AD</th><th>CONTROL</th><th>PD</th></tr></thead><tbody>${scoreRows}</tbody></table>${imageHtml}<h2>Stage timing</h2><dl>${Object.entries(row.stage_timings || {}).map(([key,value]) => `<dt>${text(key.replaceAll('_',' '))}</dt><dd>${text(Number(value).toFixed(3))} seconds</dd>`).join('')}</dl><p>Use your browser's Print command to save this report as PDF. Store exported reports privately.</p></body></html>`
}
