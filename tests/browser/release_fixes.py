"""Public-release browser regression and short Improve lifecycle.

Run Chromium from a throwaway Playwright environment, against a fresh initialized
workspace whose path contains a space. Pass its application interpreter with
--app-python; the script prepares scratch fixtures after checking the empty
import page. It clicks the real GUI for all review,
export, training, evaluation, promotion, and restore actions.
"""
import argparse
import csv
import io
import json
import re
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs,urlparse,urlencode

from playwright.sync_api import expect,sync_playwright


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--app-python',required=True,help='Application interpreter, separate from the browser environment')
    parser.add_argument('--base-url',default='http://127.0.0.1:8798')
    args=parser.parse_args()
    workspace=args.workspace.resolve()
    evidence=args.evidence.resolve()
    evidence.mkdir(parents=True,exist_ok=True)
    fixture={}
    report={'checks':{},'page_errors':[],'console_errors':[],'workspace':str(workspace),'fixture_note':'Synthetic proposals for review regression; dated variants of one bundled scene for lifecycle smoke testing only.'}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        report['browser']=browser.version
        context=browser.new_context(viewport={'width':1366,'height':900},accept_downloads=True)
        page=context.new_page()
        page.on('pageerror',lambda err:report['page_errors'].append(str(err)))
        page.on('console',lambda msg:report['console_errors'].append({'text':msg.text,'url':msg.location.get('url','')}) if msg.type=='error' else None)
        context.tracing.start(screenshots=True,snapshots=True,sources=True)

        def flush():
            (evidence/'browser-outcome.json').write_text(json.dumps(report,indent=2))
        def check(name,value):
            report['checks'][name]=value;flush();print(name,flush=True)
        def goto(path):
            page.goto(args.base_url+path,wait_until='domcontentloaded')
            expect(page.locator('h1').first).to_be_visible()
        def shot(name):
            page.screenshot(path=str(evidence/(name+'.png')),full_page=True)
            (evidence/(name+'.txt')).write_text(page.locator('body').inner_text())
        def api(path,body=None):
            response=context.request.get(args.base_url+path) if body is None else context.request.post(args.base_url+path,data=body)
            assert response.ok,(path,response.status,response.text())
            return response.json()
        def poll(path,terminal,timeout=180):
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                record=api(path)
                if terminal(record):return record
                page.wait_for_timeout(500)
            raise AssertionError(('Timed out',path,record))
        def state():
            return api('/api/state?'+urlencode({'image_id':fixture['three_box_key']}))

        try:
            goto('/')
            expect(page.locator('#modelDetails')).to_contain_text('Held-out test results: 21 images')
            expect(page.locator('#modelDetails')).to_contain_text('0.544')
            shot('01_default_model')
            goto('/model-card')
            expect(page.locator('#modelCard code').first).to_have_text('0: ship')
            expect(page.locator('#modelCard p').first).to_contain_text('This detector identifies candidate pairs')
            assert not page.locator('#modelCard').inner_text().startswith('---')
            assert page.locator('#modelCard p').count()<25
            shot('02_model_card')
            check('13_model_card','Wrapped paragraphs, inline code, escaped HTML, and no visible front matter.')

            goto('/import')
            expect(page.locator('.intake-footer')).to_contain_text('opendarksts training import-data')
            shot('02b_empty_import_public_cli_tip')
            prepared=subprocess.run([args.app_python,str(Path(__file__).with_name('release_fixes_prepare.py'))],env={**os.environ,'OPENDARKSTS_WORKSPACE':str(workspace),'PYTHONPATH':str(Path(__file__).resolve().parents[2])},capture_output=True,text=True,check=True)
            fixture=json.loads(prepared.stdout)
            goto('/detect')
            expect(page.locator('#datePeriod')).to_have_value('date')
            expect(page.locator('#end')).to_be_disabled()
            page.locator('#areaCoordinates summary').click()
            page.locator('#west').fill('-93')
            page.locator('#areaForm button[type=submit]').click()
            expect(page.locator('#areaError')).to_contain_text('West must be less than East')
            box=page.locator('#areaError').bounding_box()
            savebox=page.locator('#areaForm button[type=submit]').bounding_box()
            assert abs(box['y']-savebox['y'])<120,(box,savebox)
            shot('03_area_local_error')
            page.locator('#west').fill('-94.36')
            page.locator('#datePeriod').select_option('range')
            expect(page.locator('#end')).to_be_enabled()
            page.locator('#datePeriod').select_option('date')
            page.locator('#areaForm button[type=submit]').click()
            page.wait_for_url('**/data')
            goto('/detect')
            expect(page.locator('#datePeriod')).to_have_value('date')
            expect(page.locator('#end')).to_be_disabled()
            check('7_16_area_dates','Plain local error beside Save; Single date disables End date and survives navigation/reload.')

            goto('/manual-review?'+urlencode({'image_id':fixture['three_box_key']}))
            expect(page.get_by_role('button',name='Start Detail Review',exact=True)).to_be_visible()
            assert [box['reviewed'] for box in state()['detections']]==[False]*3
            review_posts=[]
            page.on('request',lambda request:review_posts.append({'url':request.url,'payload':request.post_data_json}) if request.method=='POST' and any(route in request.url for route in ['/api/save-detection','/api/label-detection','/api/start-detail']) else None)
            page.get_by_role('button',name='Start Detail Review',exact=True).click()
            page.wait_for_function('state && state.image.review_status === "detail"')
            page.wait_for_timeout(300)
            after_start=state()
            assert [box['reviewed'] for box in after_start['detections']]==[False]*3
            expect(page.locator('#reviewProgress')).to_contain_text('0/3')
            shot('04_start_detail_zero_reviewed')
            for i in range(3):
                page.get_by_role('button',name=re.compile(r'Accept.*Next')).click()
                page.wait_for_function('(n)=>state.counts.reviewed_active_detections===n',arg=i+1)
                page.wait_for_timeout(250)
                saved=state()
                assert [box['reviewed'] for box in saved['detections']]==[j<=i for j in range(3)],saved
                assert saved['image']['review_status']==('completed' if i==2 else 'detail')
                (evidence/f'review_after_accept_{i+1}.json').write_text(json.dumps(saved,indent=2))
                shot(f'05_accept_{i+1}_decisions')
            assert not any('/api/save-detection' in event['url'] for event in review_posts),review_posts
            saved_bytes=Path(fixture['three_box_review']).read_bytes()
            page.reload();page.wait_for_timeout(500)
            assert Path(fixture['three_box_review']).read_bytes()==saved_bytes
            (evidence/'review_after_start.json').write_text(json.dumps(after_start,indent=2))
            (evidence/'review_posts.json').write_text(json.dumps(review_posts,indent=2))
            check('1_explicit_decisions',{'after_start':[False,False,False],'after_first_accept':[True,False,False],'completed_after_decisions':3,'automatic_geometry_posts':0,'existing_saved_review_unchanged_on_reload':True})

            goto('/detect')
            page.get_by_role('button',name='Use the bundled example',exact=True).click()
            page.wait_for_url('**/data')
            page.get_by_role('link',name='Continue to Scan Images').click()
            expect(page.locator('#runButton')).to_be_enabled()
            page.locator('#runButton').click()
            page.wait_for_url('**/jobs?job=*')
            job_id=parse_qs(urlparse(page.url).query)['job'][0]
            job=poll('/api/operations/jobs/'+job_id,lambda row:row['status'] in {'completed','failed','cancelled'})
            assert job['status']=='completed',job
            expect(page.locator('#jobOutput')).to_contain_text('1 candidate detection',timeout=15000)
            log=workspace/'jobs'/job_id/'job.log'
            assert 'TIFF_Warning' not in log.read_text()
            shot('06_example_completed_job')
            page.get_by_role('link',name='Open STS Events & Live Map').click()
            expect(page.locator('#aisStatus')).to_contain_text('AIS_PRESENT')
            rows=api('/api/results?'+urlencode({'directory':job['result']['output_dir']}))['detections']
            assert len(rows)==1
            candidate=rows[0]
            assert abs(candidate['center_lat']-28.6564)<.0001 and abs(candidate['center_lon']+94.3228)<.0001
            assert abs(candidate['confidence']-.73)<.02
            shot('07_example_result')
            check('one_click_example',candidate)
            page.locator('[data-verdict=real_sts]').click()
            expect(page.locator('#verdictStatus')).to_contain_text('Real STS')
            page.locator('#improve').click()
            expect(page.locator('#improve')).to_have_text('Sent to Improve')
            transferred=api('/api/results?'+urlencode({'directory':job['result']['output_dir']}))['detections'][0]['human_review']['transfer']
            doc=json.loads(Path(transferred['review_path']).read_text())
            box=doc['detections'][0]
            assert doc['width']>=480 and doc['height']>=480,(doc['width'],doc['height'])
            assert box['width']*box['height']<.01
            assert box['review_label']=='ships_sts'
            context_pixels=doc['operational_source']['crop_pixels']
            assert abs(box['x_center']*doc['width']+context_pixels[0]-candidate['cx_px'])<.001
            assert abs(box['y_center']*doc['height']+context_pixels[1]-candidate['cy_px'])<.001
            (evidence/'transferred_review.json').write_text(json.dumps(doc,indent=2))
            with page.expect_download() as download:
                page.locator('#exports').get_by_role('link',name='CSV',exact=True).click()
            csv_download=download.value
            csv_download.save_as(evidence/csv_download.suggested_filename)
            assert csv_download.suggested_filename!='reviewed_candidates.csv'
            csv_rows=list(csv.DictReader(io.StringIO((evidence/csv_download.suggested_filename).read_text())))
            assert csv_rows[0]['review_verdict']=='real_sts' and csv_rows[0]['review_label']=='ships_sts'
            assert 'human_review' not in csv_rows[0]
            goto('/manual-review?'+urlencode({'image_id':transferred['image_key']}))
            page.wait_for_function('state && state.image')
            shot('08_context_training_image')
            check('2_11_context_transfer_exports',{'size':[doc['width'],doc['height']],'box':box,'context_pixels':context_pixels,'download':csv_download.suggested_filename})

            # Actual browser upload into a new album, with real YOLO proposals.
            goto('/import')
            page.locator('#newAlbumImport').click()
            page.locator('#namingAlbumName').fill('Release lifecycle smoke')
            with page.expect_file_chooser() as chooser:
                page.locator('[data-naming-action=pick-files]').click()
            with page.expect_response(lambda response:'/api/import/upload' in response.url and response.request.method=='POST') as uploaded:
                chooser.value.set_files(fixture['sources'])
            upload=uploaded.value.json()
            imported=poll('/api/import/jobs/'+upload['job_id'],lambda row:row['state'] in {'done','failed'})
            assert imported['state']=='done',imported
            assert imported['result']['imported_count']==3,imported
            keys=[item['image_key'] for item in imported['result']['imported']]
            for key in keys:
                goto('/manual-review?'+urlencode({'image_id':key}))
                page.wait_for_function('state && state.image')
                review_state=api('/api/state?'+urlencode({'image_id':key}))
                assert len(review_state['detections'])>=1,review_state
                page.get_by_role('button',name='Start Detail Review',exact=True).click()
                for _ in range(len(review_state['detections'])):
                    page.get_by_role('button',name=re.compile(r'Accept.*Next')).click()
                    page.wait_for_timeout(200)
                assert api('/api/state?'+urlencode({'image_id':key}))['image']['review_status']=='completed'
            shot('09_uploaded_and_annotated')

            goto('/datasets')
            expect(page.locator('#candidateBody tr')).not_to_have_count(0)
            page.wait_for_function('state.candidates.length>=5')
            for key in keys:
                page.locator('input.pick[data-key="'+key+'"]').check()
            page.locator('#datasetName').fill('release_smoke')
            page.locator('#ratioTrain').fill('34')
            page.locator('#ratioVal').fill('33')
            page.locator('#ratioTest').fill('33')
            page.locator('#saveSelection').click()
            expect(page.locator('#exportSelection')).to_be_enabled()
            page.locator('#exportSelection').click()
            expect(page.locator('#saveStatus')).to_contain_text('Exported 3 images')
            expect(page.locator('#continueToSetup')).to_be_enabled()
            selections=api('/api/dataset-selections')['selections']
            selection=next(row for row in selections if row['dataset_name']=='release_smoke')
            assert all(selection['split_counts'][split]==1 for split in ['train','val','test']),selection
            page.reload()
            expect(page.locator('#continueToSetup')).to_be_enabled()
            expect(page.locator('#datasetName')).to_have_value('release_smoke')
            shot('10_reopened_export')
            check('5_6_import_dataset_resume',{'uploaded':3,'explicitly_annotated':3,'split_counts':selection['split_counts'],'reopened_continue_enabled':True})
            page.locator('#continueToSetup').click()
            page.wait_for_url('**/training-setup?*')
            expect(page.locator('#saveConfig')).to_be_enabled()
            page.locator('#advancedSettings summary').click()
            expect(page.locator('#weightHint')).to_contain_text('opendarksts/assets/weight_recreation.md')
            for control,value in {'runGroup':'release_smoke','runName':'one_epoch','configName':'release_one_epoch','imageSize':'128','epochs':'1','batch':'1','workers':'0','device':'cpu'}.items():
                page.locator('#'+control).fill(value)
            page.locator('#epochs').press('Tab')
            expect(page.locator('#epochSummary')).to_have_text('1 epoch')
            page.locator('#augPreset').select_option('none')
            page.locator('#saveOnly').click()
            expect(page.locator('#previewCommand')).to_be_enabled()
            page.locator('#previewCommand').click()
            expect(page.locator('#commandPreview')).to_contain_text('# bash')
            preview=page.locator('#commandPreview').inner_text()
            assert "'data=" in preview and 'release fixes ' in preview,preview
            shot('11_setup_command_with_spaces')
            check('9_16_training_setup',{'recipe':'opendarksts/assets/weight_recreation.md','epoch_summary':'1 epoch','preview':preview})
            page.locator('#goTrain').click()
            page.wait_for_url('**/training?*')
            expect(page.locator('#preparedSummary')).to_contain_text('1 epoch ·')
            expect(page.locator('#startRun')).to_be_enabled()
            page.locator('#startRun').click()
            expect(page.locator('#startStatus')).to_contain_text('Started',timeout=30000)
            status=poll('/api/training/runs/release_smoke/one_epoch/status',lambda row:row['status'] in {'completed','failed','stopped'},timeout=240)
            assert status['status']=='completed',status
            goto('/evaluation?run_group=release_smoke')
            expect(page.locator('#resultsBody')).to_contain_text('one_epoch')
            expect(page.locator('#promoteWeight')).to_be_disabled()
            expect(page.locator('#promoteHelp')).to_contain_text('filename')
            page.locator('button.test-split').click()
            expect(page.locator('#testStatus')).to_contain_text('Test split complete',timeout=180000)
            page.locator('#writeReport').click()
            expect(page.locator('#reportStatus')).to_contain_text('Wrote')
            evaluation=api('/api/training/run-groups/release_smoke')['runs'][0]
            assert evaluation['test_metrics']['images']==1,evaluation
            assert evaluation['test_metrics']['split']=='test'
            page.locator('#targetName').fill('release_smoke_best.pt')
            expect(page.locator('#promotionWarning')).to_contain_text('far below')
            shot('12_before_promotion_warning')
            cancelled=[]
            def reject(dialog):cancelled.append(dialog.message);dialog.dismiss()
            page.once('dialog',reject)
            page.locator('#promoteWeight').click()
            assert cancelled and 'far below' in cancelled[0]
            assert api('/api/training/model')['is_default']
            page.once('dialog',lambda dialog:dialog.accept())
            page.locator('#promoteWeight').click()
            expect(page.locator('#promoteStatus')).to_contain_text('Promoted to')
            expect(page.locator('#modelDetails')).to_contain_text('Held-out test results: 1 image')
            promoted=api('/api/operations/model')['model']
            assert promoted['metrics']==evaluation['test_metrics'],(promoted,evaluation)
            assert 'image count not recorded' not in page.locator('#modelDetails').inner_text()
            assert f"{evaluation['test_metrics']['mAP50-95']:.3f}" in page.locator('#modelDetails').inner_text()
            shot('13_promoted_test_scores')
            goto('/')
            expect(page.locator('#modelDetails')).to_contain_text('Held-out test results: 1 image')
            shot('14_home_promoted_test_scores')
            goto('/evaluation?run_group=release_smoke')
            expect(page.locator('#groupSelect')).to_contain_text('release_smoke')
            page.set_viewport_size({'width':390,'height':844})
            picker=page.locator('#groupSelect').bounding_box()
            assert picker['width']>=250,picker
            assert picker['x']>=0 and picker['x']+picker['width']<=391,picker
            assert page.evaluate('document.documentElement.scrollWidth')<=391
            shot('15_evaluation_mobile_batch')
            page.set_viewport_size({'width':1366,'height':900})
            page.locator('#restoreDefault').click()
            expect(page.locator('#modelStatus')).to_contain_text('Restored bundled default')
            expect(page.locator('#modelDetails')).to_contain_text('21 images')
            assert api('/api/training/model')['is_default']
            shot('16_restored_default')
            check('3_4_15_training_evaluate_promote_restore',{'training':status,'evaluation':evaluation,'promoted_metrics':promoted['metrics'],'warning_cancel_kept_default':True,'mobile_picker':picker,'restored_default':True})
            assert not (workspace/'.init.lock').exists()
            assert not list((workspace/'reports/operational').glob('*/.run.lock'))
            check('16_lock_cleanup','No .init.lock or per-run .run.lock after initialization and real inference.')

            goto('/')
            page.get_by_text('Understanding AIS result labels',exact=True).click()
            expect(page.locator('body')).to_contain_text('DETECTED_ONLY (stored as AIS_MISSING)')
            expect(page.locator('body')).to_contain_text('AIS_ERROR (stored as AIS_INVALID)')
            shot('17_consistent_ais_labels')
            goto('/data')
            page.locator('#getAis').click()
            page.wait_for_url('**/jobs?job=*')
            ais_job=parse_qs(urlparse(page.url).query)['job'][0]
            expect(page.locator('#jobTitle')).to_contain_text('2020-01-24')
            cancel=page.get_by_role('button',name='Cancel download',exact=True)
            cancel.click()
            finished=poll('/api/operations/jobs/'+ais_job,lambda row:row['status'] in {'completed','failed','cancelled'})
            assert finished['status']=='cancelled',finished
            # A terminal scratch job receives a long diagnostic fixture to verify
            # actual browser scrolling without downloading a complete US AIS day.
            ais_log=workspace/'jobs'/ais_job/'job.log'
            (evidence/'ais_cancel_original.log').write_bytes(ais_log.read_bytes())
            with ais_log.open('a') as stream:
                stream.write('\n'+'\n'.join('Acceptance scrolling fixture line '+str(i) for i in range(120))+'\nAcceptance log tail marker\n')
            if not page.locator('#jobDetails').evaluate('(node)=>node.open'):
                page.locator('#jobDetails summary').click()
            page.locator('#refreshJobs').click()
            expect(page.locator('#jobLog')).to_contain_text('Acceptance log tail marker')
            scroll=page.locator('#jobLog').evaluate('(node)=>({top:node.scrollTop,height:node.scrollHeight,visible:node.clientHeight})')
            assert scroll['top']>0 and abs(scroll['top']+scroll['visible']-scroll['height'])<2,scroll
            shot('18_ais_dates_log_autoscroll')
            check('8_16_ais_labels_dates_log',{'download_status':finished['status'],'description':finished['description'],'scroll_fixture':scroll})
            report['unexpected_console_errors']=[item for item in report['console_errors'] if not (item['url'].endswith('/api/operations/project') and '400' in item['text'])]
            assert not report['unexpected_console_errors'],report['unexpected_console_errors']
            assert not report['page_errors'],report['page_errors']
        except Exception as exc:
            report['failure']=repr(exc)
            shot('FAILED')
            raise
        finally:
            flush()
            context.tracing.stop(path=str(evidence/'browser-trace.zip'))
            browser.close()
    print(json.dumps({'checks':list(report['checks']),'page_errors':report['page_errors'],'console_errors':report['console_errors']}),flush=True)


if __name__=='__main__':main()
