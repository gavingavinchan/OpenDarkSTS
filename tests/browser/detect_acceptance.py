"""Real-browser Detect acceptance regressions (requires a separate Playwright env).

Initialize a fresh workspace, run the bundled example with the foreground CLI,
then serve it on 8794. Run this script with --workspace and --evidence. The script
uses real inference and a real NOAA transfer, cancels the latter, and leaves no
active download. It deliberately does not install browser packages in the app env.
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--noaa-only', action='store_true', help='Repeat only the real-download regression in an already verified workspace')
    parser.add_argument('--base-url', default='http://127.0.0.1:8794')
    args = parser.parse_args()
    workspace, evidence = args.workspace.resolve(), args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    example = workspace / 'inputs' / 'example'
    original = example / 'AIS_2020_01_24.csv'
    renamed = example / 'my-vessel-records.csv'
    shutil.copyfile(original, renamed)
    undated = example / 'undated.csv'
    undated.write_text('MMSI,BaseDateTime,LAT,LON\n1,unknown,28,-94\n')
    checks, events = {}, []
    if args.noaa_only and (evidence / 'browser-outcome.json').exists():
        checks.update(json.loads((evidence / 'browser-outcome.json').read_text())['checks'])
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1366, 'height': 900}, accept_downloads=True)
        page = context.new_page()
        page.on('pageerror', lambda error: events.append(str(error)))

        def api(path):
            response = context.request.get(args.base_url + path)
            assert response.ok, response.text()
            return response.json()

        def goto(path):
            page.goto(args.base_url + path, wait_until='domcontentloaded')

        def shot(name):
            page.screenshot(path=str(evidence / (name + '.png')), full_page=True)
            (evidence / (name + '.txt')).write_text(page.locator('body').inner_text())

        def source(path):
            goto('/data')
            expect(page.locator('#aisPath')).not_to_have_value('')
            page.locator('#aisPath').fill(str(path))
            page.locator('#saveAis').click()
            expect(page.locator('#notice')).to_have_text('AIS source saved.')

        def scan(radius):
            goto('/run')
            expect(page.locator('#runButton')).to_be_enabled()
            page.get_by_text('Sensitivity, AIS search radius & model', exact=True).click()
            page.locator('#radius').fill(str(radius))
            expect(page.locator('#runInputs')).to_contain_text(re.compile(str(radius)[:-3] + r',?' + str(radius)[-3:] + ' metres') if radius >= 1000 else f'{radius} metres')
            page.locator('#runButton').click()
            page.wait_for_url('**/jobs?job=*')
            job_id = parse_qs(urlparse(page.url).query)['job'][0]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                record = api('/api/operations/jobs/' + job_id)
                if record['status'] in {'completed', 'failed', 'cancelled'}:
                    break
                page.wait_for_timeout(500)
            assert record['status'] == 'completed', record
            expect(page.locator('#jobTitle')).to_contain_text('completed', timeout=10000)
            return record

        try:
            if not args.noaa_only:
                # F04: the foreground example is visible before any GUI scans exist.
                goto('/')
                expect(page.locator('#detectionTotals')).to_contain_text('1 completed scan / 1 candidate detection')
                shot('F04-after-cli-home')
                goto('/jobs')
                expect(page.locator('#jobRows')).to_contain_text('radius 1000 m')
                expect(page.locator('#jobTitle')).to_contain_text('completed')
                shot('F04-after-cli-jobs')
                cli = api('/api/operations/jobs')['jobs'][0]
                rows = api('/api/results?directory=' + cli['result']['output_dir'])['detections']
                assert len(rows) == 1 and rows[0]['dark_status'] == 'AIS_PRESENT'
                assert abs(rows[0]['center_lat'] - 28.6564) < .0001
                assert abs(rows[0]['center_lon'] + 94.3228) < .0001
                checks['F04'] = {'job_id': cli['job_id'], 'candidate': rows[0]}

                goto('/detect')
                expect(page.locator('#start')).to_have_value('2020-01-24')
                page.locator('#start').fill('2020-01-24')
                page.locator('#end').fill('2020-01-24')
                page.locator('#images').fill(str(example))
                page.locator('#areaForm button[type=submit]').click()
                page.wait_for_url('**/data')
                expect(page.locator('#aisCount')).to_have_text('0 AIS files')
                page.locator('#aisPath').fill(str(renamed))
                page.locator('#saveAis').click()
                expect(page.locator('#notice')).to_have_text('AIS source saved.')
                expect(page.locator('#aisCount')).to_have_text('1 AIS file')
                shot('F07-after-source-count')
                checks['F07'] = 'Inventory changed from 0 to 1 immediately after saving.'

                goto('/run')
                expect(page.locator('#runButton')).to_be_enabled()
                page.get_by_text('Sensitivity, AIS search radius & model', exact=True).click()
                page.locator('#radius').fill('500')
                expect(page.locator('#runInputs')).to_contain_text('500 metres')
                shot('F08-after-live-radius')
                checks['F08'] = 'Unsaved 500 m radius is reflected in the confirmation summary.'

                older, newer = scan(70), scan(500)
                old_dir, new_dir = older['result']['output_dir'], newer['result']['output_dir']
                assert older['result']['summary']['PARTIALLY_DARK'] == 1
                assert newer['result']['summary']['AIS_PRESENT'] == 1
                goto('/results')
                expect(page.locator('#runs')).to_have_value(new_dir)
                page.locator('#runs').select_option(old_dir)
                expect(page.locator('#aisStatus')).to_contain_text('PARTIALLY_DARK')
                assert parse_qs(urlparse(page.url).query)['directory'] == [old_dir]
                shot('F02-after-selected-70m')
                page.reload()
                expect(page.locator('#runs')).to_have_value(old_dir)
                expect(page.locator('#aisStatus')).to_contain_text('PARTIALLY_DARK')
                shot('F02-after-refresh-70m')
                page.locator('#runs').select_option(new_dir)
                expect(page.locator('#aisStatus')).to_contain_text('AIS_PRESENT')
                page.go_back()
                expect(page.locator('#runs')).to_have_value(old_dir)
                expect(page.locator('#aisStatus')).to_contain_text('PARTIALLY_DARK')
                page.go_forward()
                expect(page.locator('#runs')).to_have_value(new_dir)
                expect(page.locator('#aisStatus')).to_contain_text('AIS_PRESENT')
                shot('F02-after-forward-500m')
                checks['F02'] = 'Refresh, back, and forward retain the selected run and matching status.'
                options = page.locator('#runs option').all_text_contents()
                assert any('radius 70 m' in text and 'threshold 0.25' in text and '2020-01-24' in text for text in options)
                assert any('radius 500 m' in text and '.tiff' in text and 'UTC' in text for text in options)
                shot('F06-after-readable-results')
                goto('/jobs?job=' + older['job_id'])
                expect(page.locator('#jobTitle')).to_contain_text('radius 70 m')
                expect(page.locator('#jobRows')).to_contain_text('radius 500 m')
                shot('F06-after-readable-jobs')
                checks['F06'] = options

                source(undated)
                goto('/run')
                expect(page.locator('#runButton')).to_be_disabled()
                expect(page.locator('#runReadiness')).to_contain_text('Cannot establish the AIS CSV date')
                shot('F03-after-undated-preflight')
                source(renamed)
                goto('/results?directory=' + new_dir)
                expect(page.locator('#aisStatus')).to_contain_text('AIS_PRESENT')
                shot('F03-after-renamed-csv-result')
                checks['F03'] = 'Renamed CSV yields AIS_PRESENT; undated contents block scanning with an explicit message.'

                page.locator('#runs').select_option(old_dir)
                expect(page.locator('#aisStatus')).to_contain_text('PARTIALLY_DARK')
                page.locator('[data-verdict=not_sts]').click()
                expect(page.locator('#negativeField')).to_be_visible()
                page.locator('#improve').click()
                expect(page.locator('#notice')).to_contain_text('human training label is required')
                shot('F09-before-label-correction')
                page.locator('#negativeLabel').select_option('ship_non_sts')
                page.locator('#improve').click()
                expect(page.locator('#improve')).to_have_text('Sent to Improve')
                expect(page.locator('#notice')).to_have_text('Reviewed crop sent to Improve.')
                shot('F09-after-success')
                checks['F09'] = 'The error is replaced by success feedback following transfer.'
                for extension in ['json', 'csv']:
                    with page.expect_download() as download:
                        page.locator('#exports').get_by_role('link', name=extension.upper(), exact=True).click()
                    result_path = evidence / ('results-reviewed.' + extension)
                    download.value.save_as(result_path)
                    goto('/jobs?job=' + older['job_id'])
                    expect(page.locator('#jobTitle')).to_contain_text('completed')
                    with page.expect_download() as download:
                        page.locator('#jobOutput a', has_text='Download ' + extension.upper()).click()
                    job_path = evidence / ('jobs-reviewed.' + extension)
                    download.value.save_as(job_path)
                    assert job_path.read_bytes() == result_path.read_bytes()
                    assert b'ship_non_sts' in job_path.read_bytes() and b'transfer' in job_path.read_bytes()
                    shot('F05-after-jobs-reviewed-export')
                    goto('/results?directory=' + old_dir)
                    expect(page.locator('#improve')).to_have_text('Sent to Improve')
                checks['F05'] = 'Actual CSV and JSON downloads from both routes are byte-identical and include review and transfer.'

            # F01: real NOAA transfer, unrelated local scan, dependency guard, cancel.
            goto('/data')
            expect(page.locator('#dateSummary')).to_have_text('2020-01-24 → 2020-01-24')
            page.locator('#getAis').click()
            page.wait_for_url('**/jobs?job=*')
            download_id = parse_qs(urlparse(page.url).query)['job'][0]
            expect(page.get_by_role('button', name='Cancel download', exact=True)).to_be_visible()
            expect(page.locator('#jobOutput')).to_contain_text(re.compile(r'Download progress: [1-9][0-9]+ / 190262962 bytes'), timeout=60000)
            first = page.locator('#jobOutput').inner_text()
            page.wait_for_timeout(3500)
            expect(page.locator('#jobOutput')).not_to_have_text(first)
            shot('F01-after-real-noaa-progress')
            source(workspace / 'inputs' / 'ais' / 'full')
            goto('/run')
            expect(page.locator('#runButton')).to_be_disabled()
            expect(page.locator('#runReadiness')).to_contain_text('download of the selected inputs')
            shot('F01-after-dependent-scan-guard')
            source(renamed)
            goto('/run')
            expect(page.locator('#runButton')).to_be_enabled()
            shot('F01-after-independent-scan-enabled')
            local = scan(1000)
            assert local['result']['summary']['AIS_PRESENT'] == 1
            assert api('/api/operations/jobs/' + download_id)['status'] == 'running'
            goto('/jobs?job=' + download_id)
            page.get_by_role('button', name='Cancel download', exact=True).click()
            expect(page.locator('#jobTitle')).to_contain_text('cancelled', timeout=30000)
            shot('F01-after-cancelled-cleanup')
            cancelled = api('/api/operations/jobs/' + download_id)
            assert cancelled['status'] == 'cancelled'
            assert not list((workspace / 'inputs' / 'ais').glob('.download-*'))
            assert not list((workspace / 'inputs' / 'ais' / 'full').glob('*.csv'))
            assert original.exists() and renamed.exists()
            checks['F01'] = {'job': cancelled, 'local_scan_job_id': local['job_id'], 'partials_removed': True}
            assert not events, events
            print(json.dumps({'checks': list(checks), 'browser': browser.version, 'page_errors': events}, indent=2))
        finally:
            # Also cancel a live NOAA job if a browser assertion failed midway.
            for job in api('/api/operations/jobs')['jobs']:
                if job['kind'] in {'ais', 'imagery'} and job['status'] not in {'completed', 'failed', 'cancelled'}:
                    context.request.post(args.base_url + '/api/operations/jobs/' + job['job_id'] + '/cancel', data={})
            (evidence / 'browser-outcome.json').write_text(json.dumps({'checks': checks, 'page_errors': events, 'browser_version': browser.version}, indent=2))
            browser.close()


if __name__ == '__main__':
    main()
