"""Actual local Chrome screenshots and keyboard checks; isolated QA browser only.

Requires installed Chrome and the already locked websockets dependency. Uses
Chrome's documented headless mode and DevTools protocol, without touching the
user's browser profile. Never generates screenshots from a template.
"""

import argparse
import base64
import json
import os
import subprocess
import uuid
from pathlib import Path
from time import monotonic, sleep
from urllib.request import urlopen

from websockets.sync.client import connect

from shiftlab.data.storage import seal, write_json
from shiftlab.experiments.provenance import file_sha256

PAGES = ("Overview", "Datasets", "Experiment Runner", "Results", "Diagnostics", "Report Exports")


class Browser:
    def __init__(self, websocket_url: str):
        self.connection = connect(websocket_url, proxy=None, max_size=20000000, legacy=True)
        self.identity = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.identity += 1
        identity = self.identity
        self.connection.send(json.dumps({"id": identity, "method": method, "params": params or {}}))
        while True:
            response = json.loads(self.connection.recv(timeout=30))
            if response.get("id") == identity:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", {})

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def wait(self, expression: str, timeout: float = 30):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            value = self.evaluate(expression)
            if value:
                return value
            sleep(0.2)
        raise RuntimeError(f"Browser condition timed out: {expression}")

    def key(self, name: str, code: int) -> None:
        for kind in ("keyDown", "keyUp"):
            self.call(
                "Input.dispatchKeyEvent",
                {
                    "type": kind,
                    "key": name,
                    "code": name,
                    "windowsVirtualKeyCode": code,
                    "nativeVirtualKeyCode": code,
                    **(
                        {"text": "\r", "unmodifiedText": "\r"}
                        if name == "Enter" and kind == "keyDown"
                        else {}
                    ),
                },
            )

    def screenshot(self, path: Path) -> None:
        data = self.call(
            "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}
        )
        path.write_bytes(base64.b64decode(data["data"]))


def review(root: Path, destination: Path, url: str) -> dict:
    if destination.exists():
        raise ValueError("Preserve the previous browser review; choose a new destination")
    destination.mkdir(parents=True)
    profile = root / "artifacts/ui_review" / f"chrome-profile-{uuid.uuid4().hex}"
    executable = Path(os.environ["ProgramFiles"]) / "Google/Chrome/Application/chrome.exe"
    log = (destination / "chrome.log").open("xb")
    process = subprocess.Popen(
        [
            str(executable),
            "--headless",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            url,
        ],
        stdout=log,
        stderr=log,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    browser = None
    try:
        deadline = monotonic() + 30
        while not (profile / "DevToolsActivePort").exists():
            if process.poll() is not None or monotonic() > deadline:
                raise RuntimeError("Owned QA Chrome did not expose its local DevTools endpoint")
            sleep(0.2)
        port = (profile / "DevToolsActivePort").read_text().splitlines()[0]
        with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as response:
            tabs = json.load(response)
        target = next(t for t in tabs if t["type"] == "page" and t["url"].startswith(url))
        browser = Browser(target["webSocketDebuggerUrl"])
        browser.call("Page.enable")
        browser.wait("document.querySelector('[data-testid=stMain] h1')?.innerText === 'Overview'")
        # Observe actual labels and then use keyboard input to traverse the radio group.
        labels = browser.wait(
            "(()=>{const labels=Array.from(document.querySelectorAll('[data-testid=stSidebar] "
            "label')).map(e=>e.innerText);return "
            + json.dumps(list(PAGES))
            + ".every(p=>labels.includes(p)) ? labels : false})()"
        )
        if not all(page in labels for page in PAGES):
            raise RuntimeError(f"Required page navigation missing: {labels}")
        browser.evaluate(
            "document.querySelector('[data-testid=stSidebar] input[type=radio]').focus()"
        )
        focused = browser.evaluate("document.activeElement.type")
        browser.key("ArrowDown", 40)
        browser.wait("document.querySelector('[data-testid=stMain] h1')?.innerText === 'Datasets'")
        keyboard = {"radio_focus": focused, "arrow_down_changed_page": True}
        browser.key("Tab", 9)
        keyboard["tab_focus_tag"] = browser.evaluate("document.activeElement.tagName")
        keyboard["tab_focus_outline"] = browser.evaluate(
            "getComputedStyle(document.activeElement).outlineStyle"
        )
        observations = []
        for width, height in ((1280, 800), (820, 1000)):
            browser.call(
                "Emulation.setDeviceMetricsOverride",
                {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
            )
            for page in PAGES:
                browser.evaluate(
                    "Array.from(document.querySelectorAll('[data-testid=stSidebar] "
                    f"label')).find(e=>e.innerText === {json.dumps(page)}).click()"
                )
                browser.wait(
                    "document.querySelector('[data-testid=stMain] h1')?.innerText === "
                    + json.dumps(page)
                )
                browser.wait("!document.querySelector('[data-testid=stStatusWidget]')")
                sleep(0.4)
                errors = browser.evaluate(
                    "Array.from(document.querySelectorAll('[data-testid="
                    "stException], [data-testid=stAlertContentError]')).map(e=>e.innerText)"
                )
                if errors:
                    raise RuntimeError(f"{page}: {errors}")
                browser.evaluate("document.querySelector('[data-testid=stMain]').scrollTop=0")
                name = f"{width}_{page.lower().replace(' ', '_')}.png"
                browser.screenshot(destination / name)
                observations.append(
                    {
                        "page": page,
                        "width": width,
                        "height": height,
                        "screenshot": name,
                        "errors": errors,
                        "body_horizontal_overflow": browser.evaluate(
                            "document.body.scrollWidth > window.innerWidth"
                        ),
                        "heading": browser.evaluate(
                            "document.querySelector('[data-testid=stMain] h1').innerText"
                        ),
                    }
                )
                print(f"Captured {page} at {width} × {height}", flush=True)
        browser.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1280, "height": 800, "deviceScaleFactor": 1, "mobile": False},
        )
        browser.evaluate(
            "Array.from(document.querySelectorAll('[data-testid=stSidebar] "
            "label')).find(e=>e.innerText === 'Results').click()"
        )
        browser.wait("document.querySelector('[data-testid=stMain] h1')?.innerText === 'Results'")
        browser.wait(
            "Array.from(document.querySelectorAll('summary')).some(e=>"
            "e.innerText.includes('Exact baseline'))"
        )
        browser.evaluate(
            "Array.from(document.querySelectorAll('summary')).find(e=>"
            "e.innerText.includes('Exact baseline')).click()"
        )
        browser.evaluate(
            "Array.from(document.querySelectorAll('[data-testid=stSelectbox]')).find("
            "e=>e.innerText.includes('Inspect comparison')).scrollIntoView()"
        )
        sleep(0.5)
        browser.screenshot(destination / "1280_result_trace.png")
        browser.evaluate(
            "Array.from(document.querySelectorAll('[data-testid=stSidebar] "
            "label')).find(e=>e.innerText === 'Diagnostics').click()"
        )
        browser.wait(
            "document.querySelector('[data-testid=stMain] h1')?.innerText === 'Diagnostics'"
        )
        for method, heading in (
            ("importance_weighting", "Source importance weights"),
            ("coral", "Saved covariance alignment"),
        ):
            browser.wait(
                "Array.from(document.querySelectorAll('[data-testid=stSelectbox]')).some("
                "e=>e.innerText.includes('Diagnostic run'))"
            )
            browser.evaluate(
                "Array.from(document.querySelectorAll('[data-testid=stSelectbox]')).find("
                "e=>e.innerText.includes('Diagnostic run')).querySelector('input').focus()"
            )
            browser.key("ArrowDown", 40)
            browser.call("Input.insertText", {"text": method})
            browser.wait("!!document.querySelector('[role=option]')")
            browser.key("Enter", 13)
            browser.wait(
                "document.querySelector('[data-testid=stMain]').innerText.includes("
                + json.dumps(heading)
                + ")"
            )
            browser.wait("!document.querySelector('[data-testid=stStatusWidget]')")
            browser.wait("Array.from(document.images).every(e=>e.complete && e.naturalWidth>0)")
            browser.evaluate("document.querySelector('[data-testid=stMain]').scrollTop=0")
            sleep(0.4)
            browser.screenshot(destination / f"1280_diagnostics_{method}.png")
        keyboard["diagnostic_select_search_enter"] = True
        browser.evaluate(
            "Array.from(document.querySelectorAll('[data-testid=stSidebar] "
            "label')).find(e=>e.innerText === 'Report Exports').click()"
        )
        browser.wait(
            "Array.from(document.querySelectorAll('button')).some("
            "e=>e.innerText === 'Build verified bundle')"
        )
        browser.evaluate(
            "Array.from(document.querySelectorAll('button')).find("
            "e=>e.innerText === 'Build verified bundle').focus()"
        )
        keyboard["export_focus"] = browser.evaluate("document.activeElement.innerText")
        browser.key("Enter", 13)
        browser.wait(
            "document.querySelector('[data-testid=stMain]').innerText.includes('Export ready')"
        )
        keyboard["enter_built_verified_export"] = True
        browser.wait("!document.querySelector('[data-testid=stStatusWidget]')")
        browser.wait("Array.from(document.images).every(e=>e.complete && e.naturalWidth>0)")
        sleep(0.5)
        browser.evaluate("document.querySelector('[data-testid=stMain]').scrollTop=0")
        browser.screenshot(destination / "1280_export_ready.png")
        receipt = seal(
            {
                "url": url,
                "browser": browser.call("Browser.getVersion"),
                "method": "Actual Chrome headless render / DevTools captureScreenshot",
                "keyboard": keyboard,
                "observations": observations,
                "app_file_hashes": {
                    p.relative_to(root).as_posix(): file_sha256(p)
                    for p in sorted((root / "app").glob("*.py"))
                },
                "screenshot_hashes": {p.name: file_sha256(p) for p in destination.glob("*.png")},
                "references": [
                    "https://developer.chrome.com/docs/automation-and-testing/headless",
                    "https://chromedevtools.github.io/devtools-protocol/",
                ],
            }
        )
        write_json(destination / "browser_review.json", receipt)
        return receipt
    except Exception:
        if browser:
            browser.screenshot(destination / "failure.png")
            write_json(
                destination / "failure.json",
                {
                    "body": browser.evaluate("document.body.innerText"),
                    "focused": browser.evaluate("document.activeElement.outerHTML"),
                },
            )
        raise
    finally:
        if browser:
            browser.call("Browser.close")
            browser.connection.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=20)
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    args = parser.parse_args()
    result = review(Path.cwd(), args.out, args.url)
    print(f"Reviewed {len(result['observations'])} actual page/viewport renders")
