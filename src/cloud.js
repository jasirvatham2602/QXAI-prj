import { createClient } from '@supabase/supabase-js'
import { previewAuth, previewApi } from './preview'
export const previewMode = import.meta.env.DEV && new URLSearchParams(window.location.search).get('preview') === '1'
const url = import.meta.env.VITE_SUPABASE_URL
const key = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || import.meta.env.VITE_SUPABASE_ANON_KEY
if (key?.startsWith('sb_secret_')) throw new Error('Use a publishable key in the frontend, never a secret key')
export const auth = previewMode ? previewAuth : url && key ? createClient(url, key) : null
const apiUrl = new URL(import.meta.env.VITE_API_URL || 'http://localhost:7860')
if (['localhost','127.0.0.1'].includes(apiUrl.hostname) && ['localhost','127.0.0.1'].includes(window.location.hostname)) apiUrl.hostname = window.location.hostname
const base = apiUrl.href.replace(/\/$/, '')
const demoAdminBase = 'https://jasirvatham2602-qxai-backend-hugging-face.hf.space'
const isLoopbackPage = ['localhost','127.0.0.1'].includes(window.location.hostname)
export async function checkReadiness() {
  if (previewMode) return true
  try {
    const response = await fetch(base + '/health/ready', { signal: AbortSignal.timeout(10000) })
    return response.ok
  } catch { return false }
}
export async function api(path, options = {}) {
  if (previewMode) return previewApi(path, options)
  if (!auth) throw new Error('Authentication is not configured')
  const { data } = await auth.auth.getSession()
  if (!data.session) throw new Error('Please sign in again')
  const response = await fetch(base + path, { ...options, headers: { ...options.headers, Authorization: `Bearer ${data.session.access_token}` } })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw Object.assign(new Error(typeof error.detail === 'string' ? error.detail : 'Request failed. Please try again.'),{status:response.status})
  }
  if (response.status === 204) return null
  return response.headers.get('content-type')?.includes('image/') ? response.blob() : response.json()
}

export async function adminApi(path,options={}) {
  if (previewMode) {
    if (path==='/admin/session') return {local_login_enabled:false,authenticated:true}
    if (path.startsWith('/admin/analyses')) {
      const result=await previewApi(path.replace('/admin',''),options)
      if (Array.isArray(result)) return result.map(row=>({...row,uploader:{id:'sample-user',email:'researcher@example.invalid',display_name:'Sample researcher'}}))
      return result
    }
  }
  const session=auth ? (await auth.auth.getSession()).data.session : null
  const isDemoLogin=path==='/admin/login'
  const targets=[base]
  if(isLoopbackPage && base!==demoAdminBase)targets.push(demoAdminBase)
  let response
  for (let index=0;index<targets.length;index++) {
    try {
      response=await fetch(targets[index]+path,{...options,credentials:'include',headers:{
        ...(options.body && !(options.body instanceof FormData)?{'Content-Type':'application/json'}:{}),
        ...(session && targets[index]===base && !isDemoLogin?{Authorization:`Bearer ${session.access_token}`} : {}),...options.headers,
      }})
      break
    } catch (error) {
      if(index===targets.length-1)throw error
    }
  }
  if(!response.ok){const data=await response.json().catch(()=>({}));throw Object.assign(new Error(typeof data.detail==='string'?data.detail:'Admin request failed'),{status:response.status})}
  if(response.status===204)return null
  return response.headers.get('content-type')?.includes('image/')?response.blob():response.json()
}

export async function modelInfo(){
  if(previewMode)return null
  try{const response=await fetch(base+'/models',{signal:AbortSignal.timeout(10000)});return response.ok?response.json():null}catch{return null}
}
