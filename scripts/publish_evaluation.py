"""Publish aggregate research metrics only; never copy images or training file paths."""
import argparse
import json
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('candidate',type=Path)
args=parser.parse_args()
report=json.loads((args.candidate/'evaluation.json').read_text())
manifest=json.loads((args.candidate/'model_manifest.json').read_text())
cleaned=args.candidate.name.startswith('cleaned-v3')
public={
    'dataset_title':report.get('dataset_title','Cleaned AD/PD v3 — dataset version 12' if cleaned else 'Original Kaggle dataset — supplied test split'),
    'test_name':report.get('test_name','cleaned-v3-seed42-random-image-holdout' if cleaned else 'supplied-kaggle-test'),
    'model_version':manifest['model_version'], 'classes':report['classes'],
    'candidate':report['candidate_test'],'original':report['old_checkpoint_test'],
    'test_images':sum(row['support'] for row in report['candidate_test']['classical']['per_class'].values()),
    'preprocessing_version':manifest['preprocessing_version'],
    'seed':report['seed'],'dataset_url':report['dataset_url'],
    'limitations':report['limitations'],
    'evaluation_note':('Locked 70/10/20 random image split; exact duplicate groups excluded across splits. Fresh candidate training and selection exclude this test set. Prior project models have overlapping source data, so saved-model metrics are diagnostic baselines. Unknown patient IDs; patient independence is not established.' if cleaned else 'Supplied test split; exact duplicate overlap removed from candidate training. Unknown patient IDs. Original training membership is unknown and source data overlaps this test set, so saved-model metrics are a diagnostic baseline.'),
}
path=Path(__file__).resolve().parents[1]/'public/model-evaluation.json'
path.write_text(json.dumps(public,indent=2)+'\n')
print('Published aggregate model evaluation; no images or individual filenames included.')
