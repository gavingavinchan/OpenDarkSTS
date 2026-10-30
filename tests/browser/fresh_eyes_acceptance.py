"""Newcomer acceptance against an initialized, freshly served workspace.

Run with Playwright/Chromium installed in a throwaway environment. This script
runs real inference through GUI controls; it does not enter any example paths.
Stop the server with the app CLI after this script finishes.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8796')
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    report = {'checks': {}, 'console_errors': [], 'page_errors': [], 'pages': []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1366, 'height': 900}, accept_downloads=True)
        page = context.new_page()
        context.on('page', lambda tab: tab.on('pageerror', lambda err: report['page_errors'].append(str(err))))
        page.on('pageerror', lambda err: report['page_errors'].append(str(err)))
        page.on('console', lambda msg: report['console_errors'].append({'url': page.url, 'text': msg.text}) if msg.type == 'error' else None)

        def goto(path):
            page.goto(args.base_url + path, wait_until='domcontentloaded')
            expect(page.locator('h1')).to_be_visible()

        def shot(name):
            page.screenshot(path=str(evidence / (name + '.png')), full_page=True)
            (evidence / (name + '.txt')).write_text(page.locator('body').inner_text())

        def api(path):
            r = context.request.get(args.base_url + path)
            assert r.ok, r.text()
            return r.json()

        try:
            goto('/')
            expect(page.locator('#modelDetails h3')).to_have_text('YOLO11s')
            expect(page.locator('#modelDetails')).to_contain_text('0.920')
            expect(page.locator('#modelDetails')).to_contain_text('Precision measures')
            shot('home_model_1366')
            with page.expect_popup() as popup:
                page.get_by_role('link', name='Open YOLO11s model card').click()
            card = popup.value
            expect(card.locator('body')).to_contain_text('21 test images')
            card.screenshot(path=str(evidence / 'model_card_1366.png'), full_page=True)
            card.close()
            goto('/evaluation')
            expect(page.locator('#modelDetails h3')).to_have_text('YOLO11s')
            expect(page.locator('#modelDetails')).to_contain_text('0.849')
            expect(page.locator('#targetName')).to_have_attribute('placeholder', 'yolo11s_myrun_2026-01-01_best.pt')
            report['checks']['model'] = 'YOLO11s, per-class table, definitions, and browser model card on Home and Step 7.'

            goto('/detect')
            expect(page.locator('#start')).to_have_value('2020-01-24')
            expect(page.locator('#start')).to_have_attribute('placeholder', 'YYYY-MM-DD')
            expect(page.locator('#images')).to_be_visible()
            page.get_by_role('button', name='Use the bundled example', exact=True).click()
            page.wait_for_url('**/data')
            expect(page.locator('#imageCount')).to_have_text('1 image file')
            expect(page.locator('#aisCount')).to_have_text('1 AIS file')
            expect(page.locator('#getImagery')).to_be_disabled()
            expect(page.locator('#credentialsStatus')).to_contain_text('restarted')
            page.get_by_text('Linux and Windows setup instructions', exact=True).click()
            expect(page.locator('#credentialsHelp')).to_contain_text('PowerShell')
            shot('example_selected_data_1366')
            project = api('/api/operations/project')
            assert project['start'] == project['end'] == '2020-01-24'
            assert Path(project['images']) == args.workspace.resolve() / 'inputs' / 'example'
            assert Path(project['ais_dir']).is_absolute()
            assert project['bbox'][0] < -94.3228 < project['bbox'][2]
            assert project['bbox'][1] < 28.6564 < project['bbox'][3]
            page.get_by_role('link', name='Continue to Scan Images').click()
            expect(page.locator('#runButton')).to_be_enabled()
            shot('example_ready_run_1366')
            page.locator('#runButton').click()
            page.wait_for_url('**/jobs?job=*')
            job_id = parse_qs(urlparse(page.url).query)['job'][0]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                job = api('/api/operations/jobs/' + job_id)
                if job['status'] in {'completed', 'failed', 'cancelled'}:
                    break
                page.wait_for_timeout(500)
            assert job['status'] == 'completed', job
            expect(page.locator('#jobOutput')).to_contain_text('1 candidate detection', timeout=10000)
            expect(page.locator('#jobLog')).not_to_contain_text('TIFF_Warning')
            shot('example_completed_job_1366')
            page.get_by_role('link', name='Open STS Events & Live Map').click()
            expect(page.locator('#aisStatus')).to_contain_text('AIS_PRESENT')
            expect(page.locator('#confidenceValue')).to_contain_text('0.73')
            expect(page.locator('#resultReading')).to_contain_text('not classified as dark')
            expect(page.locator('#vessels')).to_contain_text('Tanker (AIS code 80)')
            expect(page.locator('#vessels')).not_to_contain_text('unknown m')
            expect(page.locator('.candidate-marker')).to_have_attribute('fill', 'none')
            assert page.locator('.ais-marker').count() == 2
            rows = api('/api/results?' + page.url.split('?')[1])['detections']
            assert len(rows) == 1
            result = rows[0]
            assert result['dark_status'] == 'AIS_PRESENT' and result['ais_count'] == 2
            assert abs(result['center_lat'] - 28.6564) < .0001
            assert abs(result['center_lon'] + 94.3228) < .0001
            report['checks']['example'] = {'job_id': job_id, 'candidate': result, 'project': project}
            shot('example_results_1366')
            page.locator('.candidate-marker').click(position={'x':13,'y':1}, force=True)
            expect(page.locator('.leaflet-popup')).to_contain_text('Inspect candidate 1')
            shot('live_map_popup_1366')
            page.locator('.leaflet-popup-close-button').click()
            page.locator('[data-verdict=real_sts]').click()
            expect(page.locator('#verdictStatus')).to_contain_text('Real STS')
            page.locator('#improve').click()
            expect(page.locator('#improve')).to_have_text('Sent to Improve')
            expect(page.locator('#reviewHelp')).to_contain_text('locked')
            expect(page.locator('[data-verdict=real_sts]')).to_be_disabled()
            with page.expect_download() as download:
                page.locator('#exports').get_by_role('link', name='CSV', exact=True).click()
            download.value.save_as(evidence / 'reviewed_candidates.csv')
            assert str(args.workspace.resolve()) in (evidence / 'reviewed_candidates.csv').read_text()
            report['checks']['review'] = 'Plain verdict labels; locked transfer explained; actual reviewed CSV downloaded.'

            paths = ['/', '/detect', '/data', '/run', '/results', '/jobs', '/import', '/manual-review', '/bbox-overview', '/datasets', '/training-setup', '/training', '/evaluation', '/model-card']
            for width, height in [(1366, 900), (1920, 1080), (390, 844)]:
                page.set_viewport_size({'width': width, 'height': height})
                for path in paths:
                    goto(path)
                    page.wait_for_timeout(500)
                    # The first import is now populated by the reviewed candidate.
                    if path == '/import':
                        page.get_by_role('button', name='Image list', exact=True).click()
                        expect(page.locator('.ledger-table')).to_be_visible()
                    if path == '/results':
                        expect(page.locator('#confidenceValue')).to_contain_text('0.73')
                    overflow = page.evaluate('({page:document.documentElement.scrollWidth, viewport:innerWidth})')
                    assert overflow['page'] <= width + 1, (path, width, overflow)
                    rail = page.locator('.rail')
                    if width == 390 and rail.count():
                        for stage in rail.locator('a').all():
                            box = stage.bounding_box()
                            assert box['x'] >= 0 and box['x'] + box['width'] <= width + 1, (path, box)
                    if path == '/results' and width == 390:
                        verdict = page.locator('#candidates td[data-label="Image verdict"]')
                        expect(verdict).to_contain_text('Real STS')
                        box = verdict.bounding_box()
                        assert box['x'] >= 0 and box['x'] + box['width'] <= width
                    name = path.strip('/') or 'home'
                    shot(f'{name}_{width}x{height}')
                    report['pages'].append({'path': path, 'width': width, 'height': height, 'title': page.title(), 'overflow': overflow})
            assert not report['page_errors'], report['page_errors']
            assert not report['console_errors'], report['console_errors']
            report['checks']['responsive'] = 'All 14 pages at all three sizes: no page overflow or clipped training tabs; no console or page errors.'
        finally:
            report['browser'] = browser.version
            (evidence / 'browser-outcome.json').write_text(json.dumps(report, indent=2))
            browser.close()
    print(json.dumps({'checks': list(report['checks']), 'pages': len(report['pages']), 'console_errors': report['console_errors'], 'page_errors': report['page_errors']}, indent=2))


if __name__ == '__main__':
    main()
