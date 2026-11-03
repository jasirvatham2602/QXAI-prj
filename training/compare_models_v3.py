"""Paired analysis of fixed checkpoints; no training or model selection."""
import json, math
import torch
import numpy as np
from scipy.stats import binomtest, chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_v3 import OUT, load_pair, extract_features
from train import metrics

def heatmap(matrix, names_x, names_y, title, xlabel, ylabel, filename):
    fig,ax=plt.subplots(figsize=(7,5))
    plotted=ax.imshow(matrix,cmap='Blues',aspect='auto')
    fig.colorbar(plotted,ax=ax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else '#222')
    ax.set(xticks=range(len(names_x)),yticks=range(len(names_y)),xticklabels=names_x,yticklabels=names_y,xlabel=xlabel,ylabel=ylabel,title=title)
    ax.set_xticks(np.arange(-.5,len(names_x),1),minor=True);ax.set_yticks(np.arange(-.5,len(names_y),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.5);ax.tick_params(which='minor',bottom=False,left=False)
    ax.tick_params(axis='y',labelrotation=90);ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14);ax.title.set_size(18)
    fig.tight_layout();fig.savefig(OUT/filename,dpi=180);plt.close(fig)

def main():
    torch.set_num_threads(4)
    manifest=json.loads((OUT/'split_manifest.json').read_text())
    expected=json.loads((OUT/'evaluation.json').read_text())
    cnn,q=load_pair(OUT,torch.device('cpu'))
    features,y=extract_features(cnn,manifest['splits']['test'],torch.device('cpu'))
    with torch.no_grad():
        cp=cnn.classifier(features).argmax(1).numpy()
        qp=torch.cat([q(x) for x in features.split(128)]).argmax(1).numpy()
    truth=y.numpy();cc=cp==truth;qc=qp==truth
    table=np.array([[sum(qc&cc),sum(qc&~cc)],[sum(~qc&cc),sum(~qc&~cc)]],dtype=int)
    b,c=int(table[0,1]),int(table[1,0]);n=b+c
    statistic=(abs(b-c)-1)**2/n if n else 0
    p_exact=float(binomtest(b,n,.5).pvalue) if n else 1.
    report={'test_name':manifest['test_name'],'n':len(truth),'table_rows':'QXAI correct, incorrect','table_columns':'XAI correct, incorrect','contingency':table.tolist(),'discordant_qxai_only_correct':b,'discordant_xai_only_correct':c,'exact_two_sided_p':p_exact,'continuity_corrected_chi2':statistic,'continuity_corrected_p':float(chi2.sf(statistic,1)),'accuracy_difference_xai_minus_qxai':float(cc.mean()-qc.mean()),'independence_limitation':'Image-level pairs; subject clustering is unknown. Non-significance does not establish equivalence.'}
    for kind,pred,name,title in [('classical',cp,'cm_XAI.png','Confusion Matrix for classical XAI model'),('quantum',qp,'cm_QXAI.png','Confusion Matrix for QXAI model')]:
        result=metrics(truth,pred)
        assert result['confusion_matrix']==expected['candidate_test'][kind]['confusion_matrix']
        heatmap(np.array(result['confusion_matrix']).T,['AD','CONTROL','PD'],['AD','CONTROL','PD'],title,'Actual Label','Predicted Label',name)
    heatmap(table,['Correct','Incorrect'],['Correct','Incorrect'],"McNemar’s Contingency Matrix\nbetween QXAI and XAI",'XAI','QXAI','Contingency_Matrix.png')
    (OUT/'mcnemar.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'paired_test_predictions.json').write_text(json.dumps([{'image_sha256':row['group'],'true':int(t),'xai':int(a),'qxai':int(b)} for row,t,a,b in zip(manifest['splits']['test'],truth,cp,qp)],indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
