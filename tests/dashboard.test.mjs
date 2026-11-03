import test from 'node:test'
import assert from 'node:assert/strict'
import { historyQuery, visibleRows, reportHtml, reportDownloadName } from '../src/dashboard-utils.mjs'

const record = {
  id: '<script>alert(1)</script>', status: 'complete', created_at: '2026-11-03T05:00:00Z',
  result: { classical: { label: 'CONTROL', probabilities: [0.1,0.8,0.1] }, quantum: { label: 'AD', probabilities: [0.6,0.3,0.1] } },
}
test('date filters use inclusive local calendar days across DST changes', () => {
  const query = new URLSearchParams(historyQuery({start:'2026-11-01',end:'2026-11-01',page:2}).split('?')[1])
  assert.equal(query.get('start'), '2026-11-01T04:00:00.000Z')
  assert.equal(query.get('end'), '2026-11-02T05:00:00.000Z')
  assert.equal(query.get('offset'), '100')
})
test('disagreement filter excludes queued jobs without results', () => {
  assert.deepEqual(visibleRows([{id:'queued',status:'queued'},record],'disagreement'),[record])
})
test('report escapes record values and refuses executable image URLs', () => {
  const html = reportHtml(record, { input:'javascript:alert(1)', saliency:'data:image/png;base64,aGVsbG8=' })
  assert.ok(html.includes('&lt;script&gt;alert(1)&lt;/script&gt;'))
  assert.ok(!html.includes('<script>'))
  assert.ok(!html.includes('javascript:'))
  assert.ok(html.includes('data:image/png;base64,aGVsbG8='))
  assert.ok(html.includes('The models disagree'))
  assert.ok(html.includes('80.0%'))
})

test('PDF export produces a real PDF document', async () => {
  const {createPdfReport}=await import('../src/pdf-report.mjs')
  const blob=createPdfReport({...record,uploader:{display_name:'Test researcher',email:'test@example.invalid'}})
  const bytes=new Uint8Array(await blob.arrayBuffer())
  assert.equal(new TextDecoder().decode(bytes.slice(0,5)),'%PDF-')
  assert.ok(bytes.length>1000)
})

test('HTML report includes and escapes the original filename',()=>{
  const html=reportHtml({...record,original_filename:'AD_<script>.png'})
  assert.ok(html.includes('Original filename'))
  assert.ok(html.includes('AD_&lt;script&gt;.png'))
  assert.ok(reportHtml(record).includes('Not recorded for this older upload'))
})

test('download names identify the source scan and avoid unsafe characters',()=>{
  assert.equal(reportDownloadName({id:'1234abcd-0000',original_filename:'PD_001.png'},'pdf'),'qxai-PD_001-1234abcd.pdf')
  assert.ok(!reportDownloadName({id:'1',original_filename:'../AD<script>.png'},'html').includes('<'))
})
