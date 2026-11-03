import { jsPDF } from 'jspdf'
import { modelsDisagree } from './dashboard-utils.mjs'

export function createPdfReport(row, images = {}) {
  const pdf = new jsPDF({unit:'pt',format:'a4',compress:true})
  const width=pdf.internal.pageSize.getWidth(), height=pdf.internal.pageSize.getHeight()
  const margin=42, usable=width-margin*2
  let y=margin
  pdf.setProperties({title:'QXAI research analysis',subject:'Model comparison and saliency evidence',creator:'QXAI'})
  function space(required) { if(y+required>height-48){pdf.addPage();y=margin} }
  function text(value,size=11,bold=false,color=[30,52,67]) {
    pdf.setFont('helvetica',bold?'bold':'normal');pdf.setFontSize(size);pdf.setTextColor(...color)
    const lines=pdf.splitTextToSize(String(value ?? 'Not recorded'),usable)
    for(const line of lines){space(size*1.5);pdf.text(line,margin,y);y+=size*1.45}
    y+=4
  }
  function title(value){space(45);y+=12;text(value,15,true)}
  text('QXAI / RESEARCH REPORT',11,true,[20,92,112])
  text('Model analysis and comparison',24,true)
  text('Research demonstration. Model predictions are not clinical diagnoses. Probability scores are not calibrated clinical confidence.',10,false,[85,99,108])
  if(row.preview_note)text(row.preview_note,10)
  title('Analysis record')
  for(const [label,value] of [
    ['Original filename',row.original_filename || 'Not recorded for this older upload'],
    ['Analysis ID',row.id],['Created',row.created_at],['Status',row.status],
    ['Model version',row.model_version],['Preprocessing',row.preprocessing_version],
    ['Processing time',row.duration_seconds==null?'Not recorded':`${row.duration_seconds.toFixed(2)} seconds`],
    ['Uploader',row.uploader?.display_name],['Uploader email',row.uploader?.email],['Uploader ID',row.uploader?.id],
  ]){if(value!=null)text(`${label}: ${value}`)}
  title('Classical and quantum predictions')
  text(modelsDisagree(row)?'The models disagree on this sample.':'The models agree on this sample.',11,true)
  for(const kind of ['classical','quantum']){
    const model=row.result?.[kind]
    if(!model)continue
    text(`${kind==='classical'?'Classical XAI':'Quantum XAI'}: ${model.label}`,12,true)
    text(model.probabilities.map((p,i)=>`${['AD','CONTROL','PD'][i]}: ${(p*100).toFixed(1)}%`).join('     '))
  }
  if(Object.keys(row.stage_timings || {}).length){
    title('Processing breakdown')
    for(const [key,value] of Object.entries(row.stage_timings))text(`${key.replaceAll('_',' ')}: ${Number(value).toFixed(3)} seconds`)
  }
  for(const kind of ['input','saliency']){
    const data=images[kind]
    if(!/^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=]+$/.test(data||''))continue
    const props=pdf.getImageProperties(data)
    const imageWidth=Math.min(usable,kind==='input'?300:usable)
    const imageHeight=Math.min(imageWidth*props.height/props.width,height-150)
    const scaledWidth=imageHeight*props.width/props.height
    space(imageHeight+58)
    title(kind==='input'?'Input sample':'Saliency comparison')
    pdf.addImage(data,props.fileType,margin,y,scaledWidth,imageHeight)
    y+=imageHeight+16
  }
  const pages=pdf.getNumberOfPages()
  for(let page=1;page<=pages;page++){
    pdf.setPage(page);pdf.setFontSize(9);pdf.setTextColor(100,110,120)
    pdf.text('QXAI research - store exported reports privately',margin,height-24)
    pdf.text(`${page} / ${pages}`,width-margin,height-24,{align:'right'})
  }
  return pdf.output('blob')
}
