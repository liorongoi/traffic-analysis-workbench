"""Opens the new-mission screen of the running LanePulse in a browser and checks that the SAM 3.1 detector can be
chosen there (not greyed out), as a person would see it. Saves a screenshot.
    python mac_sam_screen.py <url> <screenshot.png>
"""
import sys

from playwright.sync_api import sync_playwright

url, shot = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1280, "height": 1000})
    page.goto(url.rstrip("/") + "/#/new")
    choice = page.locator("label.choice", has_text="SAM 3.1")
    choice.wait_for(timeout=30000)
    page.screenshot(path=shot, full_page=True)
    text = choice.inner_text()
    radio = choice.locator("input[type=radio]")
    print(text)
    if radio.is_disabled():
        sys.exit(f"The SAM 3.1 option is greyed out: {text}")
    radio.check()
    print("SAM 3.1 can be chosen on the new-mission screen.")
    b.close()
