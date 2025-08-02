import asyncio
from playwright.async_api import async_playwright

# CONFIGURE HERE
import sys

DEALER_ID = sys.argv[1]
SYNDICATOR_NAME = sys.argv[2]
INVENTORY_TYPES_TO_ENABLE = sys.argv[3].split(",")

TYPE_SUFFIXES = {
    "Usagé": "_use",
    "Neuf": "_new",
    "Démonstrateur": "_demo",
}

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
        name = await checkbox.get_attribute("name")
        current_checked = await checkbox.is_checked()

        should_check = False
        if name.endswith("_export"):
            should_check = True
        else:
            for label, suffix in TYPE_SUFFIXES.items():
                if name.endswith(suffix) and label in INVENTORY_TYPES_TO_ENABLE:
                    should_check = True

        if should_check and not current_checked:
            await checkbox.check()

    await page.click("input[type=submit][value='Save']")
    print(f"✅ Exports enabled for {SYNDICATOR_NAME} at Dealer {DEALER_ID}: {INVENTORY_TYPES_TO_ENABLE}")
    await browser.close()

async def main():
    async with async_playwright() as playwright:
        await run(playwright)

asyncio.run(main())
