"""Reproducible frozen-backbone training, duplicate-safe source split, and held-out comparison."""
import argparse
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import transforms
from models import ClassicalCNN,QuantumHybrid,CLASSES

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS=Path(os.environ.get('QXAI_IMAGENET_WEIGHTS',str(Path(torch.hub.get_dir())/'checkpoints/efficientnet_b0_rwightman-7f5810bc.pth')))
MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def metrics(labels,predictions):
    matrix=np.zeros((3,3),dtype=int)
    for true,pred in zip(labels,predictions):matrix[int(true),int(pred)]+=1
    counts=matrix.sum(1);predicted=matrix.sum(0);correct=np.diag(matrix)
    recall=np.divide(correct,counts,out=np.zeros(3,dtype=float),where=counts>0)
    precision=np.divide(correct,predicted,out=np.zeros(3,dtype=float),where=predicted>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(3,dtype=float),where=(precision+recall)>0)
    return {'accuracy':float(correct.sum()/max(1,matrix.sum())),'balanced_accuracy':float(recall.mean()),
            'macro_f1':float(f1.mean()),'confusion_matrix':matrix.tolist(),
            'per_class':{label:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(counts[i])} for i,label in enumerate(CLASSES)}}


def make_split(archive,output,seed):
    start=time.monotonic();rows=[];invalid=[]
    with zipfile.ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            parts=Path(name).parts
            if '__MACOSX' in parts or Path(name).name.startswith('._') or Path(name).suffix.lower() not in ('.png','.jpg','.jpeg'):continue
            label=parts[-2]
            if label not in CLASSES:continue
            try:
                with Image.open(io.BytesIO(source.read(name))) as image:
                    image=image.convert('RGB')
                    canonical=image.resize((224,224),Image.Resampling.BILINEAR)
                    digest=hashlib.sha256(canonical.tobytes()).hexdigest()
                    rows.append({'path':name,'label':label,'target':CLASSES.index(label),'group':digest,
                                 'source_split':'test' if '/test/' in '/'+name else 'train','size':list(image.size)})
            except Exception as error:invalid.append({'path':name,'error_type':type(error).__name__})
    groups=collections.defaultdict(list)
    for row in rows:groups[row['group']].append(row)
    conflicts={group for group,members in groups.items() if len({row['label'] for row in members})>1}
    selected=[];train_test_leakage=0;duplicate_removed=0
    for group,members in sorted(groups.items()):
        if group in conflicts:continue
        test=[row for row in members if row['source_split']=='test']
        chosen=(test or members)[0]
        if test:train_test_leakage+=sum(row['source_split']=='train' for row in members)
        duplicate_removed+=len(members)-1
        selected.append(chosen)
    rng=random.Random(seed)
    splits={'train':[],'validation':[],'test':[]}
    for label in CLASSES:
        training=[row for row in selected if row['label']==label and row['source_split']=='train']
        test=[row for row in selected if row['label']==label and row['source_split']=='test']
        rng.shuffle(training)
        validation_count=max(1,round(len(training)*0.15))
        splits['validation']+=training[:validation_count]
        splits['train']+=training[validation_count:]
        splits['test']+=test
    for split in splits:splits[split]=sorted(splits[split],key=lambda row:row['path'])
    manifest={'seed':seed,'split_policy':'Preserve supplied Kaggle test membership; test wins cross-split duplicate groups; stratified 15% validation from remaining train groups.',
              'grouping':'SHA256 of decoded RGB resized to 224x224 with PIL bilinear interpolation',
              'patient_independence':'Not established: no patient/subject identifiers are provided.',
              'source_images':len(rows),'retained_images':len(selected),'contradictory_groups_excluded':len(conflicts),
              'contradictory_images_excluded':sum(len(groups[g]) for g in conflicts),
              'duplicate_images_removed':duplicate_removed,'train_images_removed_for_test_overlap':train_test_leakage,
              'invalid_images':invalid,'counts':{split:dict(collections.Counter(row['label'] for row in samples)) for split,samples in splits.items()},'splits':splits}
    for left in splits:
        for right in splits:
            if left!=right:assert not ({r['group'] for r in splits[left]} & {r['group'] for r in splits[right]})
    (output/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'event':'split_ready','seconds':time.monotonic()-start,**{k:v for k,v in manifest.items() if k not in ('splits','invalid_images')}}),flush=True)
    return manifest

class ArchiveDataset(Dataset):
    def __init__(self,archive,rows,normalize):
        self.archive=zipfile.ZipFile(archive);self.rows=rows
        operations=[transforms.Resize((224,224)),transforms.ToTensor()]
        if normalize:operations.append(transforms.Normalize(MEAN,STD))
        self.transform=transforms.Compose(operations)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(io.BytesIO(self.archive.read(row['path']))) as image:
            return self.transform(image.convert('RGB')),row['target']


def extract(model,archive,rows,normalize,batch_size=32):
    loader=DataLoader(ArchiveDataset(archive,rows,normalize),batch_size=batch_size,shuffle=False,num_workers=0)
    xs=[];ys=[];start=time.monotonic()
    model.eval()
    with torch.no_grad():
        for index,(images,labels) in enumerate(loader):
            xs.append(model(images,return_features=True));ys.append(labels)
            if (index+1)%25==0:print(json.dumps({'event':'features','images':min((index+1)*batch_size,len(rows)),'total':len(rows),'seconds':time.monotonic()-start}),flush=True)
    return torch.cat(xs),torch.cat(ys)


def predict(model,x,batch=256):
    model.eval()
    with torch.no_grad():return torch.cat([model(part).argmax(1) for part in x.split(batch)]).numpy()


def train_head(kind,model,features,labels,val_x,val_y,epochs,seed,output):
    seed_all(seed)
    counts=torch.bincount(labels,minlength=3).float()
    weights=len(labels)/(3*counts)
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001 if kind=='classical' else 0.003,weight_decay=0.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=5,factor=0.5)
    loader=DataLoader(TensorDataset(features,labels),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=-1;stale=0;history=[];best_state=None
    for epoch in range(1,epochs+1):
        start=time.monotonic();model.train();total_loss=0
        for x,y in loader:
            optimizer.zero_grad();loss=loss_fn(model(x),y);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step();total_loss+=float(loss.detach())*len(y)
        result=metrics(val_y.numpy(),predict(model,val_x))
        score=result['macro_f1']
        scheduler.step(score)
        row={'epoch':epoch,'loss':total_loss/len(labels),'validation':result,'seconds':time.monotonic()-start}
        history.append(row)
        print(json.dumps({'event':'epoch','kind':kind,**row}),flush=True)
        if score>best+1e-6:
            best=score;stale=0;best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:stale+=1
        if stale>=12:break
    model.load_state_dict(best_state);model.eval()
    (output/(kind+'_history.json')).write_text(json.dumps(history,indent=2))
    return {'best_validation_macro_f1':best,'epochs_run':len(history),'selected_epoch':max(history,key=lambda row:row['validation']['macro_f1'])['epoch']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,default=ROOT/'training/data/kaggle.zip');parser.add_argument('--output',type=Path,default=ROOT/'models/candidates/kaggle-v1-seed183');parser.add_argument('--seed',type=int,default=183);parser.add_argument('--classical-epochs',type=int,default=80);parser.add_argument('--quantum-epochs',type=int,default=70);parser.add_argument('--backbone-weights',type=Path,default=DEFAULT_WEIGHTS);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True);seed_all(args.seed)
    start=time.monotonic();manifest=make_split(args.archive,output,args.seed)
    state=torch.load(args.backbone_weights,map_location='cpu')
    cnn=ClassicalCNN(state).eval()
    combined=manifest['splits']['train']+manifest['splits']['validation']+manifest['splits']['test']
    cache=ROOT/('training/data/features-normalized-seed%s.pt'%args.seed)
    fingerprint=hashlib.sha256(json.dumps({'rows':combined,'backbone_sha256':hashlib.sha256(args.backbone_weights.read_bytes()).hexdigest(),'mean':MEAN,'std':STD},sort_keys=True).encode()).hexdigest()
    if cache.exists():
        cached=torch.load(cache,map_location='cpu')
        if cached['fingerprint']!=fingerprint:raise RuntimeError('Feature cache does not match manifest')
        features,labels=cached['features'],cached['labels']
    else:
        features,labels=extract(cnn,args.archive,combined,True)
        torch.save({'features':features,'labels':labels,'fingerprint':fingerprint},cache)
    ntrain=len(manifest['splits']['train']);nval=len(manifest['splits']['validation'])
    train_x,val_x,test_x=features[:ntrain],features[ntrain:ntrain+nval],features[ntrain+nval:]
    train_y,val_y,test_y=labels[:ntrain],labels[ntrain:ntrain+nval],labels[ntrain+nval:]
    print(json.dumps({'event':'feature_cache_ready','counts':[len(train_y),len(val_y),len(test_y)],'seconds':time.monotonic()-start}),flush=True)
    # Only the head trains; the exported CNN carries the exact frozen ImageNet buffers.
    classical_info=train_head('classical',cnn.classifier,train_x,train_y,val_x,val_y,args.classical_epochs,args.seed,output)
    torch.save(cnn.state_dict(),output/'xai_model.pth')
    seed_all(args.seed+1);quantum=QuantumHybrid()
    quantum_info=train_head('quantum',quantum,train_x,train_y,val_x,val_y,args.quantum_epochs,args.seed+1,output)
    torch.save(quantum.state_dict(),output/'qxai_model.pth')
    # The supplied held-out test is evaluated only after validation checkpoint selection.
    candidate={'classical':metrics(test_y.numpy(),predict(cnn.classifier,test_x)), 'quantum':metrics(test_y.numpy(),predict(quantum,test_x))}
    old_cnn=ClassicalCNN();old_cnn.load_state_dict(torch.load(ROOT/'backend/xai_model.pth',map_location='cpu'));old_cnn.eval()
    old_q=QuantumHybrid();old_q.load_state_dict(torch.load(ROOT/'backend/qxai_model.pth',map_location='cpu'));old_q.eval()
    old_features,old_labels=extract(old_cnn,args.archive,manifest['splits']['test'],False)
    baseline={'classical':metrics(old_labels.numpy(),predict(old_cnn.classifier,old_features)), 'quantum':metrics(old_labels.numpy(),predict(old_q,old_features))}
    report={'dataset_url':'https://www.kaggle.com/datasets/farjanakabirsamanta/alzheimer-diseases-3-class','dataset_api_url':'https://www.kaggle.com/api/v1/datasets/download/farjanakabirsamanta/alzheimer-diseases-3-class',
            'archive_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),'source':'Fresh Kaggle public API download','seed':args.seed,'classes':CLASSES,
            'candidate_preprocessing':{'rgb':True,'resize':[224,224],'interpolation':'torchvision PIL bilinear','to_tensor':True,'normalize_mean':MEAN,'normalize_std':STD,'version':'resize224-rgb-imagenet-normalized-v2'},
            'baseline_preprocessing':{'rgb':True,'resize':[224,224],'to_tensor':True,'normalization':False},
            'backbone':'torchvision EfficientNet-B0 ImageNet1K_V1; parameters and BatchNorm buffers frozen/eval',
            'checkpoint_selection':'Validation macro F1; no test metrics used for selection','training':{'classical':classical_info,'quantum':quantum_info},
            'candidate_test':candidate,'old_checkpoint_test':baseline,'test_counts':dict(collections.Counter(row['label'] for row in manifest['splits']['test'])),
            'limitations':['No patient IDs; patient independence cannot be established.','Exact duplicate groups excluded across splits; near duplicates and acquisition confounds can remain.','PD may differ in acquisition modality from AD/CONTROL.','Old checkpoint training provenance is unknown; old model may have seen some test images.','Dataset license metadata says Other (specified in description); redistribution rights not established.'],
            'runtime_seconds':time.monotonic()-start,'versions':{'torch':torch.__version__,'numpy':np.__version__}}
    report['checkpoint_sha256']={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('xai_model.pth','qxai_model.pth')}
    (output/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'event':'completed','report':report}),flush=True)

if __name__=='__main__':main()
