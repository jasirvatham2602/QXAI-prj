// Local design preview only. No authentication or backend requests are performed.
import exampleSaliency from '../readmeImages/sidebysideSaliencyMap.png'

export const previewSession = {
  user: { id: 'local-preview', email: 'preview@example.invalid', app_metadata: { role: 'qxai_admin' } },
}
export const previewAuth = {
  auth: {
    getSession: async () => ({ data: { session: previewSession } }),
    onAuthStateChange: () => ({ data: { subscription: { unsubscribe() {} } } }),
  },
}
const records = [
  {
    id: 'sample-complete', original_filename: 'PD_research_sample.png', created_at: '2026-11-03T05:20:00Z', status: 'complete',
    result: {
      classical: { label: 'PD', probabilities: [0.08, 0.12, 0.80] },
      quantum: { label: 'PD', probabilities: [0.12, 0.15, 0.73] },
    },
    duration_seconds: 12.4, model_version: 'sample-version', attempts: 1, preprocessing_version: 'resize224-rgb-totensor-v1', stage_timings: { classical_seconds: 1.1, quantum_seconds: 1.3, saliency_seconds: 9.0 },
    preview_note: 'Illustrative scores and timing. The saliency image is an example from the original project, not an output of this sample record.',
  },
  {
    id: 'sample-disagreement', original_filename: 'AD_research_sample.png', created_at: '2026-11-03T05:10:00Z', status: 'complete',
    result: {
      classical: { label: 'CONTROL', probabilities: [0.20, 0.67, 0.13] },
      quantum: { label: 'AD', probabilities: [0.55, 0.35, 0.10] },
    },
    duration_seconds: 15.8, model_version: 'sample-version',
    preview_note: 'Illustrative record to demonstrate the model-disagreement view.',
  },
  { id: 'sample-processing', created_at: '2026-11-03T05:25:00Z', status: 'processing' },
  { id: 'sample-queued', created_at: '2026-11-03T05:26:00Z', status: 'queued' },
  { id: 'sample-failed', created_at: '2026-11-03T05:00:00Z', status: 'failed', failure_category: 'sample_failure' },
]
export async function previewApi(path, options = {}) {
  if (options.method && options.method !== 'GET') throw new Error('Sample preview is read-only. Connect Supabase to submit or delete analyses.')
  if (path.startsWith('/analyses?')) {
    const query = new URLSearchParams(path.split('?')[1])
    const filtered = records.filter(row => (!query.get('start') || row.created_at >= query.get('start')) && (!query.get('end') || row.created_at < query.get('end')))
    const offset = Number(query.get('offset') || 0)
    return structuredClone(filtered.slice(offset, offset + Number(query.get('limit') || 50)))
  }
  if (path === '/analyses') return structuredClone(records)
  if (path === '/operations') return { queued: 1, processing: 1, complete: 2, failed: 1, average_seconds: 14.1, p95_seconds: 15.63, ready_workers: 1, oldest_queue_seconds: 20, failures_last_24h: 1 }
  if (path.endsWith('/images/saliency')) return (await fetch(exampleSaliency)).blob()
  if (path.endsWith('/images/input')) return new Blob([
    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="180"><rect width="640" height="180" fill="#eef3f5"/><text x="320" y="85" text-anchor="middle" font-family="sans-serif" fill="#526675" font-size="20">Sample scan placeholder</text><text x="320" y="115" text-anchor="middle" font-family="sans-serif" fill="#526675" font-size="14">No scan has been uploaded in this local preview.</text></svg>',
  ], { type: 'image/svg+xml' })
  throw new Error('Preview record not found')
}
