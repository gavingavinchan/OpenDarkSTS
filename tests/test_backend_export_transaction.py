"""Export failures must preserve the last usable dataset for GUI and CLI users."""
import json
from pathlib import Path

import pytest

from _helpers import detection, write_review, write_selection
from scripts import training_pipeline_lib as training


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def ready_export(tmp_path):
    reviews, cache, dataset = tmp_path / 'reviews', tmp_path / 'cache', tmp_path / 'dataset'
    review = write_review(reviews, cache, reviewer='tester', image_key='first', detections=[detection('one', 'ships_sts')])
    selection = write_selection(tmp_path / 'selection.json', reviews_dir=reviews, output_dir=dataset,
                                items=[{'image_key': 'first', 'split': 'train'}])
    training.export_dataset_selection(selection)
    (dataset / 'notes.txt').write_text('Locally maintained dataset notes')
    return selection, review, dataset


@pytest.mark.parametrize('failure', ['missing_review', 'missing_cache', 'invalid_split', 'invalid_background'])
def test_failed_reexport_keeps_previous_dataset(tmp_path, failure):
    selection, review, dataset = ready_export(tmp_path)
    before = snapshot(dataset)
    if failure == 'missing_cache':
        Path(json.loads(review.read_text())['cache_path']).unlink()
    else:
        doc = json.loads(selection.read_text())
        doc['items'][0]['split'] = 'val'
        if failure == 'missing_review':
            doc['items'].append({'image_key': 'missing', 'split': 'val'})
        elif failure == 'invalid_split':
            doc['items'][0]['split'] = 'not_a_split'
        else:
            doc['background_negatives'] = {'enabled': True, 'mode': 'invalid'}
        selection.write_text(json.dumps(doc))
    with pytest.raises((ValueError, FileNotFoundError)):
        training.export_dataset_selection(selection)
    assert snapshot(dataset) == before


def test_publish_failure_rolls_back_existing_export(tmp_path, monkeypatch):
    selection, _, dataset = ready_export(tmp_path)
    before = snapshot(dataset)
    replace = Path.replace
    failed = False

    def interrupted_replace(source, target):
        nonlocal failed
        if not failed and Path(target) == dataset / 'data.yaml' and source.parent != dataset:
            failed = True
            raise OSError('Simulated filesystem publish failure')
        return replace(source, target)

    monkeypatch.setattr(Path, 'replace', interrupted_replace)
    with pytest.raises(OSError, match='publish failure'):
        training.export_dataset_selection(selection)
    assert failed
    assert snapshot(dataset) == before


def test_export_publishes_final_yaml_and_removes_previous_split_files(tmp_path):
    selection, _, dataset = ready_export(tmp_path)
    doc = json.loads(selection.read_text())
    doc['items'][0]['split'] = 'val'
    selection.write_text(json.dumps(doc))
    manifest = training.export_dataset_selection(selection)
    assert not list((dataset / 'images' / 'train').iterdir())
    assert (dataset / 'images' / 'val' / 'first.png').is_file()
    assert (dataset / 'notes.txt').read_text() == 'Locally maintained dataset notes'
    assert (dataset / 'data.yaml').read_text().splitlines()[0] == f'path: {dataset}'
    assert all(training.file_hash(dataset / item['path']) == item['sha256'] for item in manifest['exported_files'])


def test_reexport_refuses_linked_output_and_preserves_link_target(tmp_path):
    selection, _, dataset = ready_export(tmp_path)
    shared = tmp_path / 'shared_images'
    (dataset / 'images').replace(shared)
    (dataset / 'images').symlink_to(shared, target_is_directory=True)
    before = snapshot(shared)
    with pytest.raises(ValueError, match='symbolic link'):
        training.export_dataset_selection(selection)
    assert snapshot(shared) == before
    assert (dataset / 'images').is_symlink()
