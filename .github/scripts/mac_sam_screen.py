"""Opens the new-mission screens of the running LanePulse in a browser and checks that the SAM 3.1 detector can be
chosen there (not greyed out), as a person would see it: on the one-screen New mission page it must not say it is not
installed, and on the step-by-step setup its option must be selectable. Saves a screenshot of the step-by-step setup.
    python mac_sam_screen.py <url> <screenshot.png>
"""
import re
import sys

from playwright.sync_api import sync_playwright

url, shot = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1280, "height": 1000})
    page.goto(url.rstrip("/") + "/#/new")
    quick = page.get_by_role("checkbox", name=re.compile(r"SAM 3\.1"))
    quick.wait_for(timeout=30000)
    page.wait_for_function("() => !document.body.innerText.includes('Loading')", timeout=30000)
    page.wait_for_timeout(1000)
    if "Not installed here" in quick.inner_text():
        sys.exit("The New mission page says SAM 3.1 is not installed.")
    page.goto(url.rstrip("/") + "/#/new/full")
    # Detectors are tick boxes; match the one whose text starts with SAM 3.1.
    choice = page.locator("label.choice", has_text=re.compile(r"^\s*SAM 3\.1"))
    choice.wait_for(timeout=30000)
    page.screenshot(path=shot, full_page=True)
    text = choice.inner_text()
    radio = choice.locator("input[type=checkbox]")
    print(text)
    if radio.is_disabled():
        sys.exit(f"The SAM 3.1 option is greyed out: {text}")
    radio.check()
    print("SAM 3.1 can be chosen on the new-mission screen.")
    b.close()
