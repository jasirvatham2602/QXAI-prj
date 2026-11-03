"""Storage path validation shared by user deletion and opt-in retention."""
import uuid

def analysis_prefix(owner, analysis_id):
    return f'{uuid.UUID(str(owner))}/{uuid.UUID(str(analysis_id))}'

def object_path(owner, analysis_id, kind, output_path=None):
    prefix = analysis_prefix(owner, analysis_id)
    path = output_path if kind == 'saliency' and output_path else f'{prefix}/{kind}.png'
    if not path.startswith(prefix + '/') or '..' in path or '/' in path[len(prefix)+1:]:
        raise ValueError('Invalid analysis object path')
    return path
