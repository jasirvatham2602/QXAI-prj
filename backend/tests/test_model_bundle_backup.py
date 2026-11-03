import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from model_bundle import load_bundle

def test_explicit_bundle_cannot_silently_use_legacy_preprocessing(tmp_path):
    with pytest.raises(ValueError,match='requires a manifest'):load_bundle(tmp_path)

def test_bundle_refuses_wrong_class_order_before_loading_weights(tmp_path):
    (tmp_path/'model_manifest.json').write_text(json.dumps({'class_names':['CONTROL','AD','PD'],'image_size':224}))
    with pytest.raises(ValueError,match='class order'):load_bundle(tmp_path)

def test_encrypted_backup_authentication_and_recovery():
    crypto=pytest.importorskip('cryptography.fernet')
    from backup import encode_archive,decode_archive
    owner='00000000-0000-0000-0000-000000000001';aid='00000000-0000-0000-0000-000000000002'
    rows=[{'id':aid,'owner_id':owner,'status':'failed'}]
    key=crypto.Fernet.generate_key();files={f'images/{owner}/{aid}/input.png':b'private sample'}
    encrypted=encode_archive(rows,files,key)
    assert b'private sample' not in encrypted
    manifest,recovered=decode_archive(encrypted,key)
    assert manifest['records']==1 and recovered[next(iter(files))]==b'private sample'
    with pytest.raises(crypto.InvalidToken):decode_archive(encrypted,crypto.Fernet.generate_key())
    with pytest.raises(crypto.InvalidToken):decode_archive(encrypted[:-2]+b'00',key)

def test_bundle_rejects_tampered_checkpoint(tmp_path):
    manifest={'class_names':['AD','CONTROL','PD'],'image_size':224,
      'preprocessing_version':'resize224-rgb-totensor-v1','mean':None,'std':None,
      'model_version':'test','checkpoints':{'classical':{'file':'xai_model.pth','sha256':'0'*64},'quantum':{'file':'qxai_model.pth','sha256':'0'*64}}}
    (tmp_path/'xai_model.pth').write_bytes(b'tampered checkpoint')
    (tmp_path/'model_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='checksum'):load_bundle(tmp_path)

def test_bundle_rejects_checkpoint_path_escape(tmp_path):
    manifest={'class_names':['AD','CONTROL','PD'],'image_size':224,
      'preprocessing_version':'resize224-rgb-totensor-v1','mean':None,'std':None,
      'model_version':'test','checkpoints':{'classical':{'file':'../secret.pth','sha256':'0'*64},'quantum':{'file':'qxai_model.pth','sha256':'0'*64}}}
    (tmp_path/'model_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='filename'):load_bundle(tmp_path)
