"""Opening/focusing proposed boxes must never accept their machine labels."""
import copy
import json
import re
import shutil
import subprocess

import pytest

from _helpers import detection, write_json, write_review
from scripts import review_visual_detections_ui as ui, visual_review_lib as review
from test_training_pipeline_ui import UiServer, get_json, post_json


def test_start_detail_then_accept_once_only_decides_first_box(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'REVIEWS_DIR', tmp_path/'reviews')
    monkeypatch.setattr(ui, 'REVIEWER', 'tester')
    monkeypatch.setattr(ui, 'MANIFEST_PATH', tmp_path/'manifest.json')
    monkeypatch.setattr(ui, 'refresh_detection_crop', lambda *args: None)
    boxes = [detection(f'auto-{i}', 'ships_sts', x_center=.2*i, width=.04312345) for i in range(1, 4)]
    for box in boxes:
        box.update(reviewed=False, reviewed_at=None, review_label=None)
    path = write_review(tmp_path/'reviews', tmp_path/'cache', reviewer='tester', image_key='three', detections=boxes, review_status='overview_1')
    write_json(tmp_path/'manifest.json', {'images': {}})
    ui.invalidate_review_queue_cache()
    with UiServer(ui.ReviewHandler) as server:
        def state():
            return get_json(server.base_url+'/api/state?image_id=three')
        def post(route, **payload):
            return post_json(server.base_url+'/api/'+route, {'image_id': 'three', **payload})
        initial = state()
        current = post('start-detail')
        assert current['counts']['reviewed_active_detections'] == 0
        assert current['current_detection_id'] == 'auto-1'
        # Reproduce the former automatic geometry save (including display rounding).
        for i in range(3):
            box = current['detections'][i]
            geometry = {k: round(box[k], 6) for k in ('x_center', 'y_center', 'width', 'height')}
            current = post('save-detection', detection_id=box['detection_id'], **geometry)
            assert current['counts']['reviewed_active_detections'] == i
            if i == 0:
                assert current['image']['review_status'] == 'detail'
            current = post('label-detection', detection_id=box['detection_id'], review_label='ships_sts', class_id=1, **geometry)
            assert current['counts']['reviewed_active_detections'] == i+1
            assert [d['reviewed'] for d in current['detections']] == [j <= i for j in range(3)]
            assert current['image']['review_status'] == ('completed' if i == 2 else 'detail')
        persisted = path.read_bytes()
        assert state()['image']['review_status'] == 'completed'
        assert path.read_bytes() == persisted
    ui.invalidate_review_queue_cache()


def test_automatic_geometry_save_preserves_existing_decisions_and_timestamp():
    for label in ('ships_sts', 'ship_non_sts', 'neither'):
        box = detection('old', label)
        doc = {'detections': [box]}
        before = copy.deepcopy(doc)
        review.update_detection_geometry(doc, 'old', .5, .5, .25, .2, class_id=1)
        assert doc == before
    doc = {'detections': [dict(detection('new', 'ships_sts'), reviewed=False, reviewed_at=None, review_label=None)]}
    review.update_detection_geometry(doc, 'new', .6, .5, .25, .2, user_edited=True)
    assert doc['detections'][0]['reviewed'] is True
    assert doc['detections'][0]['review_label'] == 'ships_sts'


def test_controller_ignores_display_rounding_and_counts_only_decisions():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for controller test')
    html = ui.render_app_html(None)
    names = ('autoSaveGeometryIfNeeded', 'geometryMatchesDetection', 'getCycleReviewedCount')
    functions = []
    for name in names:
        match = re.search(r'    (?:async )?function '+name+r'\(.*?(?=\n    (?:async )?function )', html, re.S)
        assert match
        functions.append(match.group())
    harness = r'''
const assert = require('assert/strict');
let selectedDetectionId = 'one';
let state = {current_detection_id:'one',image:{image_id:'three'},counts:{active_detections:3,reviewed_active_detections:0}};
let calls = [];
let layer = {detection:{x_center:.2,y_center:.3,width:.04312345,height:.1,review_label:null},getBounds:()=>({})};
const findLayerByDetectionId=()=>layer;
const boundsToGeometry=()=>({x_center:.2,y_center:.3,width:.043124,height:.1});
const postJson=async (route,payload)=>calls.push({route,payload});
(async()=>{
 await autoSaveGeometryIfNeeded(); assert.equal(calls.length,0);
 assert.equal(getCycleReviewedCount(),0);
 layer.userEdited=true;
 await autoSaveGeometryIfNeeded(); assert.equal(calls.length,1);
 assert.equal(calls[0].payload.user_edited,true);
 await autoSaveGeometryIfNeeded(); assert.equal(calls.length,1);
 state.counts.reviewed_active_detections=1;
 assert.equal(getCycleReviewedCount(),1);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run([node, '-e', '\n'.join(functions)+harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
