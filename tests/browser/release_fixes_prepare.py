"""Prepare scratch-only regression fixtures; never rewrite an existing review.

The three-box fixture has synthetic machine proposals. The training smoke test
uses three dated variants of the bundled scene, solely to exercise all splits;
its scores must not be interpreted as an independent scientific evaluation.
"""
import json
from pathlib import Path

from PIL import Image
from scripts import training_pipeline_lib as training, visual_review_lib as review
from scripts.workspace import WORKSPACE_DIR

root=WORKSPACE_DIR/'inputs'/'acceptance fixtures'
root.mkdir(parents=True,exist_ok=True)
with Image.open(WORKSPACE_DIR/'inputs/example/2020-01-24_example.tiff') as image:
    rgb=image.convert('RGB')
    for i,day in enumerate(['2020-01-25','2020-01-26','2020-01-27']):
        variant=rgb.copy()
        variant.putpixel((0,0),(i+1,2,3))
        variant.save(root/f'{day}_smoke.png')
    rgb.save(root/'2020-01-24_three_proposals.png')
result=training.import_training_data([root/'2020-01-24_three_proposals.png'],album_name='Synthetic review regression',auto_detect=False)
key=result['imported'][0]['image_key']
path=review.review_file_path(key,review.DEFAULT_REVIEWER)
doc=review.load_review_document(path)
assert not doc['detections'] and doc['review_status']=='overview_1'
doc['detections']=[{'detection_id':f'auto-{i+1:04d}','source_type':'auto','parent_detection_id':None,
    'class_id':1 if i==0 else 0,'class_name':'ships' if i==0 else 'ship',
    'confidence':.8,'x_center':.25+i*.2,'y_center':.47854321,'width':.05321498,'height':.03143157,
    'reviewed':False,'reviewed_at':None,'review_label':None,'source_models':['synthetic regression fixture']}
    for i in range(3)]
review.save_review_document(doc,path)
fixture={'three_box_key':key,'three_box_review':str(path),'sources':[str(root/f'{day}_smoke.png') for day in ['2020-01-25','2020-01-26','2020-01-27']]}
(WORKSPACE_DIR/'acceptance_fixtures.json').write_text(json.dumps(fixture,indent=2))
print(json.dumps(fixture))
