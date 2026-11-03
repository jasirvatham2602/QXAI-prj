"""Isolated model invocation. Stdout and errors never expose uploaded filenames."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from PIL import Image
import inference


def run(source, output):
    inference.SALIENCY_DIR = output
    timings = {}
    result, image = inference.analyze(source, timings=timings)
    shutil.move(image, Path(output) / 'saliency.png')
    (Path(output) / 'result.json').write_text(json.dumps(result))
    (Path(output) / 'timings.json').write_text(json.dumps(timings))
    return result

if __name__ == '__main__':
    if sys.argv[1:] == ['--check']:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'synthetic.png'
            Image.new('RGB', (224, 224), (96, 96, 96)).save(source)
            result = run(source, directory)
            assert set(result) == {'classical', 'quantum'}
            for model in result.values():
                assert len(model['probabilities']) == 3
                assert abs(sum(model['probabilities']) - 1) < 0.001
    else:
        run(*sys.argv[1:])
