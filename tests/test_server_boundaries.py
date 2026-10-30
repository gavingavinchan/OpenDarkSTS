import http.client
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest
from PIL import Image

from scripts import training_pipeline_ui as ui, training_pipeline_lib as training, visual_review_lib as review
from test_training_pipeline_ui import UiServer, post_json_capture


def request(server, *, headers, body=b'{}', path='/api/operations/project', method='POST'):
    url = urlparse(server.base_url)
    client = http.client.HTTPConnection(url.hostname, url.port)
    client.request(method, path, body=body, headers=headers)
    response = client.getresponse()
    result = response.status, response.read()
    client.close()
    return result


def test_server_rejects_foreign_origin_host_and_non_json_requests():
    with UiServer(ui.TrainingPipelineHandler) as server:
        for headers, expected in [({'Origin':'https://foreign.example','Content-Type':'application/json'},403),
                                   ({'Host':'rebound.example','Content-Type':'application/json'},403),
                                   ({'Content-Type':'text/plain'},415),
                                   ({'Content-Type':'application/json','Content-Length':str(2*1024*1024)},413)]:
            assert request(server, headers=headers)[0] == expected
        assert request(server, headers={'Content-Type':'application/json'}, body=b'[]')[0] == 400


@pytest.mark.parametrize('name', ['../bad.png','/absolute.png','C:\\outside.png','folder/../../bad.png'])
def test_upload_paths_reject_traversal(name):
    with pytest.raises(ValueError):
        ui.safe_relative_path(name)


def test_upload_rejects_disguised_non_image(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'IMPORT_STAGING_ROOT', tmp_path/'staging')
    with pytest.raises(ValueError, match='Invalid image'):
        ui.start_import_job([('bad.png',b'not a picture')],force=False,album_name=None,album_description=None,source_display_name=None)
    assert not list((tmp_path/'staging').rglob('*.png'))


def test_export_cannot_delete_outside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path/'isolated'
    outside = tmp_path/'external'
    sentinel = outside/'images/train/keep.txt'
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('keep')
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'schema_version':1,'sources':{'reviews_dir':str(tmp_path/'reviews')},'export':{'output_dir':str(outside)},'items':[]}))
    monkeypatch.setattr(training, 'WORKSPACE_DIR', workspace)
    with pytest.raises(ValueError, match='inside'):
        training.export_dataset_selection(selection)
    assert sentinel.read_text() == 'keep'


def test_cache_resource_uses_custom_workspace_and_rejects_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'WORKSPACE_DIR', tmp_path)
    target = tmp_path/'cache'/'folder with spaces'/'image.png'
    target.parent.mkdir(parents=True)
    Image.new('RGB',(2,2)).save(target)
    url = review.cache_url_for(target)
    assert '%20' in url
    assert review.resolve_cache_resource(url.removeprefix('/cache/')) == target
    assert review.resolve_cache_resource('workspace/cache/../../secret') is None
    with pytest.raises(ValueError):
        review.review_file_path('../../outside','default',tmp_path/'reviews')


def test_saved_export_output_path_is_checked_by_http(tmp_path, monkeypatch):
    outside = tmp_path.parent.parent.parent/'not-workspace'
    with UiServer(ui.TrainingPipelineHandler) as server:
        status, result = post_json_capture(server.base_url+'/api/dataset-selections/export', {'selection_path':str(outside/'selection.json')})
        assert status == 400
        assert "inside" in result["error"]
