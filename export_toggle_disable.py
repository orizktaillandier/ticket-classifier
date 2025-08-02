import asyncio
from playwright.async_api import async_playwright

# CONFIGURE HERE
import sys

DEALER_ID = sys.argv[1]
SYNDICATOR_NAME = sys.argv[2]

# This version DEACTIVATES all export checkboxes for a syndicator

async def run(playwright):
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(storage_state="auth.json")
    page = await context.new_page()

    url = f"https://aaadmin.d2cmedia.ca/administration/exportationbydealer.php?id={DEALER_ID}"
    await page.goto(url)
    print(f"🔍 Loaded export settings for dealer {DEALER_ID}")

    await page.wait_for_selector("table")
    rows = await page.query_selector_all("table tr")
    target_row = None

    for row in rows:
        cells = await row.query_selector_all("td")
        if len(cells) < 2:
            continue
        label_raw = await cells[1].inner_text()
        label = label_raw.replace("✓", "").strip().lower()
        if SYNDICATOR_NAME.lower() in label:
            target_row = row
            break

    if not target_row:
        print(f"❌ Could not find row for: {SYNDICATOR_NAME}")
        await browser.close()
        return

    checkboxes = await target_row.query_selector_all("input[type=checkbox]")
    for checkbox in checkboxes:
        current_checked = await checkbox.is_checked()
        if current_checked:
            await checkbox.uncheck()

    await page.click("input[type=submit][value='Save']")
    print(f"✅ All exports deactivated for {SYNDICATOR_NAME} at Dealer {DEALER_ID}")
    await browser.close()

async def main():
    async with async_playwright() as playwright:
        await run(playwright)

asyncio.run(main())
