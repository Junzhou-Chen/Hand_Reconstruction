#!/usr/bin/env python3
"""Download the official public HO3D v2 ZIP in the remote data Pod.

Use the author's anonymous OneDrive UI, without local dataset transfer,
credentials, mirrors or inspecting browser cookies. Only the named ZIP is
selected. This is acquisition, not admission to the training manifest.
The archive includes evaluation files; preprocessing must use train.txt only.
Install Playwright in an isolated download runtime, not the training env.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import zipfile


OFFICIAL_README = 'https://github.com/shreyashampali/ho3d'
OFFICIAL_SHARE = 'https://1drv.ms/f/c/11742dd40d1cbdc1/ElPb2rhOCeRMg-dFSM3iwO8B5nS1SgnQJs9F6l28G0pKKg?e=TMuxgr'
ARCHIVE = 'HO3D_v2.zip'


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Dataset acquisition runs only in the authorized remote Pod')
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.download.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    temporary = root / '.browser_download'
    temporary.mkdir(exist_ok=True)
    archive = root / ARCHIVE
    state = {'dataset': 'ho3d', 'version': 'v2', 'phase': 'initializing',
             'started_utc': utc(), 'official_readme': OFFICIAL_README,
             'official_share': OFFICIAL_SHARE, 'archive': ARCHIVE,
             'version_policy': 'v2 candidate consistent with paper 66K; HaWoR exact version unpublished',
             'training_allowed_yet': False, 'author_checksum_available': False}
    mutex = threading.Lock()

    def save(**changes):
        with mutex:
            state.update(changes, updated_utc=utc())
            pending = root / 'download_status.tmp.json'
            pending.write_text(json.dumps(state, indent=2) + '\n')
            pending.replace(root / 'download_status.json')

    stopped = threading.Event()

    def progress():
        while not stopped.wait(10):
            # No Playwright calls on this thread; its API is not thread-safe.
            files = list(temporary.glob('*'))
            sizes = []
            for path in files:
                try:
                    if path.is_file():
                        sizes.append(path.stat().st_size)
                except FileNotFoundError:
                    pass  # Chromium can rename a partial file between stats.
            save(present_bytes=max(sizes, default=0))

    save()
    try:
        if not archive.exists():
            from playwright.sync_api import sync_playwright
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, downloads_path=str(temporary))
                try:
                    context = browser.new_context(accept_downloads=True, locale='en-US',
                                                  viewport={'width': 1440, 'height': 1000})
                    page = context.new_page()
                    page.goto(OFFICIAL_SHARE, wait_until='domcontentloaded', timeout=60000)
                    file_button = page.get_by_role('button', name=ARCHIVE, exact=True)
                    file_button.wait_for(state='attached', timeout=90000)
                    save(phase='selecting_official_archive', page_title=page.title())
                    # OneDrive's filename span can be clipped in its grid;
                    # select the corresponding visible row, not the text span.
                    page.get_by_role('row').filter(has=file_button).click(timeout=60000)
                    with page.expect_download(timeout=120000) as event:
                        page.get_by_role('menuitem', name='Download', exact=True).click()
                    download = event.value
                    if download.suggested_filename != ARCHIVE:
                        download.cancel()
                        raise ValueError('Official UI returned an unexpected download filename')
                    save(phase='downloading', suggested_filename=download.suggested_filename)
                    thread = threading.Thread(target=progress, daemon=True)
                    thread.start()
                    try:
                        partial = root / (ARCHIVE + '.partial')
                        download.save_as(str(partial))
                        partial.replace(archive)
                    finally:
                        stopped.set()
                        thread.join()
                finally:
                    browser.close()
        save(phase='checking', present_bytes=archive.stat().st_size)
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
            if len(names) != len(set(names)) or not any(name.endswith('train.txt') for name in names):
                raise ValueError('Archive does not contain unique entries and the official training list')
            bad = package.testzip()
            if bad is not None:
                raise ValueError(f'Archive CRC failed: {bad}')
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        save(phase='complete', sha256=digest.hexdigest(), zip_crc_passed=True,
             archive_members=len(names), completed_utc=utc())
        print(json.dumps(state), flush=True)
    except Exception as error:
        save(phase='failed', error_type=type(error).__name__, message=str(error)[:1500])
        raise
    finally:
        stopped.set()


if __name__ == '__main__':
    main()
