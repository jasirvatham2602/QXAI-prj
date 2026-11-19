import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Icon } from './Icons'
import { publicApi } from './cloud'
import vqcDiagram from '../VQC.png'
import xaiMatrix from '../models/candidates/cleaned-v3-seed42/cm_XAI.png'
import qxaiMatrix from '../models/candidates/cleaned-v3-seed42/cm_QXAI.png'
import './ProjectGuide.css'

const figures = [
  { id: 'vqc', title: 'Original VQC diagram', caption: 'Original conceptual illustration. It shows Ry encoding; the current implementation uses default X rotations, shown in the code-aligned architecture diagram.', src: vqcDiagram },
  { id: 'efficientnet-b0', title: 'EfficientNet-B0 feature extractor', caption: 'Simplified view of the convolutional stem, MBConv stages, 1280-channel projection, and pooled image features used by QXAI.', src: '/guide-figures/efficientnet-b0.svg' },
  { id: 'vqc-architecture', title: 'QXAI VQC architecture', caption: 'The implemented four-qubit hybrid path, from 1280 image features through encoding, entangling layers, measurement, and the classical output head.', src: '/guide-figures/vqc-architecture.svg' },
  { id: 'model-pipeline', title: 'Classical vs hybrid model pipeline', caption: 'Both classifier paths share EfficientNet-B0 features; the hybrid path adds a classically simulated VQC.', src: '/guide-figures/model-pipeline.svg' },
  { id: 'xai-confusion', title: 'Classical XAI confusion matrix', caption: 'Per-class classification counts for the classical model on the 407-image held-out test set.', src: xaiMatrix },
  { id: 'qxai-confusion', title: 'QXAI confusion matrix', caption: 'Per-class classification counts for the hybrid model on the same 407-image held-out test set.', src: qxaiMatrix },
]

const topicCards = [
  { group: 'Models', icon: 'chip', items: [
    { label: 'EfficientNet-B0', prompt: 'Show the EfficientNet-B0 architecture and explain its role.' },
    { label: 'Full model pipeline', prompt: 'Show the full QXAI model pipeline.' },
    { label: 'VQC architecture', prompt: 'Show the VQC architecture and explain each stage.' },
  ] },
  { group: 'Results', icon: 'chart', items: [
    { label: 'Evaluation metrics', prompt: 'What do the test set accuracy, balanced accuracy, and macro F1 mean?' },
    { label: 'Confusion matrices', prompt: 'Show the confusion matrices for both models and explain how to read them.' },
    { label: 'Statistical comparison', prompt: 'What does the McNemar test say about the model comparison?' },
    { label: 'Limitations', prompt: 'What are the limitations of these results?' },
  ] },
  { group: 'Software', icon: 'layers', items: [
    { label: 'Frontend and API', prompt: 'How does the React frontend communicate with the FastAPI backend?' },
    { label: 'Cloud deployment', prompt: 'How is QXAI deployed and what services does it use?' },
    { label: 'Guide RAG', prompt: 'How does the Project Guide retrieve sources and figures for an answer?' },
  ] },
]

const welcomeMessage = { id: 'welcome', role: 'assistant', text: 'What would you like to understand? Ask about QXAI’s models, results, VQC, or software architecture. I’ll search the reviewed public project sources and link the material used.' }

function ChatMessage({ message }) {
  return <article className={`guide-message ${message.role}`}>
    <span className="guide-avatar" aria-hidden="true">{message.role === 'assistant' ? <Icon name="spark" size={20}/> : <Icon name="user" size={20}/>}</span>
    <div className="guide-bubble">
      <div className="guide-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown></div>
      {message.figures?.map(figure => <figure className="guide-figure" key={figure.id}><img src={figure.src} alt={figure.title}/><figcaption><strong>{figure.title}</strong><span>{figure.caption}</span></figcaption></figure>)}
      {!!message.sources?.length && <div className="guide-sources"><span>Sources</span>{message.sources.map((source, index) => <a key={`${source.citation || index}-${source.url}`} href={source.url} target="_blank" rel="noreferrer"><Icon name="external" size={14}/>{source.citation ? `[${source.citation}] ` : ''}{source.label}</a>)}</div>}
    </div>
  </article>
}

export default function ProjectGuide() {
  const [messages, setMessages] = useState([welcomeMessage])
  const [draft, setDraft] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [llmStatus, setLlmStatus] = useState('checking')
  const inputRef = useRef(null)

  useEffect(() => {
    let active = true
    publicApi('/guide/status').then(status => {
      if (active) setLlmStatus(status.available ? 'ready' : 'not-configured')
    }).catch(() => { if (active) setLlmStatus('offline') })
    return () => { active = false }
  }, [])

  async function submitQuestion(value = draft) {
    const question = value.trim()
    if (!question) { setNotice('Enter a question first.'); inputRef.current?.focus(); return }
    setNotice('')
    setDraft('')
    const questionId = `${Date.now()}-${Math.random().toString(36).slice(2)}`
    setMessages(current => [...current, { id: `${questionId}-q`, role: 'user', text: question }])
    setBusy(true)
    try {
      const response = await publicApi('/guide/chat', { method: 'POST', body: JSON.stringify({ question }) })
      const responseFigures = (response.figures || []).map(figure => ({ ...figure, src: figures.find(item => item.id === figure.id)?.src })).filter(figure => figure.src)
      setMessages(current => [...current, { id: `${questionId}-a`, role: 'assistant', text: response.answer, sources: response.sources || [], figures: responseFigures }])
    } catch (error) {
      setMessages(current => [...current, { id: `${questionId}-error`, role: 'assistant', text: error.message }])
      if (/not configured/i.test(error.message)) setLlmStatus('not-configured')
      else if (/could not reach|backend/i.test(error.message)) setLlmStatus('offline')
    } finally { setBusy(false) }
  }

  const statusText = {
    checking: 'Connecting to the QXAI guide backend…',
    ready: 'Live LLM connected · Answers are grounded in retrieved public QXAI sources.',
    'not-configured': 'Backend connected · Add the server-side LLM settings to enable live answers.',
    offline: 'Guide backend unavailable · Start the local API and configure its LLM settings.',
  }[llmStatus]

  return <main className="research-shell guide-page">
    <header className="guide-page-header">
      <div className="guide-eyebrow">QXAI PROJECT GUIDE</div>
      <h1>Ask about QXAI</h1>
      <p>Explore the science and software behind the research workspace.</p>
      <div className={`guide-prototype ${llmStatus}`} role="status"><Icon name={llmStatus === 'ready' ? 'check' : 'info'} size={17}/>{statusText}</div>
    </header>

    <div className="guide-layout">
      <section className="guide-chat-card" aria-label="QXAI project guide chat">
        <div className="guide-messages" aria-live="polite">
          {messages.map(message => <ChatMessage key={message.id} message={message}/>)}
        </div>
        <div className="guide-composer-wrap">
          {notice && <span className="guide-input-notice" role="status">{notice}</span>}
          <form className="guide-composer" onSubmit={event => { event.preventDefault(); submitQuestion() }}>
            <Icon name="chat" size={21}/>
            <input ref={inputRef} value={draft} onChange={event => setDraft(event.target.value)} placeholder="Ask a project question…" aria-label="Ask a question about the QXAI project" />
            <button type="submit" aria-label="Send question" title="Send question" disabled={busy}><Icon name={busy ? 'spark' : 'send'} size={19}/></button>
          </form>
          {busy && <div className="guide-thinking" role="status"><span className="guide-spinner"/> Searching project sources and preparing an answer…</div>}
          <p><Icon name="info" size={15}/> Answers use public project sources. No scan interpretation or medical advice.</p>
        </div>
      </section>

      <aside className="guide-topics" aria-label="Suggested questions">
        <header><Icon name="spark" size={27}/><div><h2>Question ideas</h2><p>Choose a question to explore the project.</p></div></header>
        {topicCards.map(group => <section className="guide-topic-group" key={group.group}>
          <h3><Icon name={group.icon} size={18}/>{group.group}</h3>
          {group.items.map(item => <button key={item.label} onClick={() => submitQuestion(item.prompt)} disabled={busy}><span><Icon name="chat" size={16}/>{item.label}</span><Icon name="arrow" size={16}/></button>)}
        </section>)}
        <div className="guide-image-note"><Icon name="image" size={17}/><span>Choose a model or results question to retrieve its matching architecture or evaluation figure.</span></div>
      </aside>
    </div>
  </main>
}
