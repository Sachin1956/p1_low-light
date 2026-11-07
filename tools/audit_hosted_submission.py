"""Measure upload-to-download latency through the hosted Gradio API."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
from gradio_client import Client, handle_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    client = Client(args.url, verbose=False, download_files=False, httpx_kwargs={"timeout": 45})
    fixtures = sorted((ROOT / "data/handoff_fixtures").glob("*.png"))
    if args.smoke:
        samples = [fixtures[1]]
    else:
        inputs = sorted((ROOT / "data/submission_lolv2/Input").glob("*.png"))
        if len(inputs) != 100:
            raise ValueError("The full audit requires 100 available test inputs to select 20 evenly spaced IDs.")
        samples = inputs[::5]

    def restore(path):
        return client.predict(handle_file(path), "hybrid_guarded", True, True, api_name="/restore")

    origin = urlsplit(args.url).netloc

    def download(file):
        url = file["url"]
        if urlsplit(url).netloc != origin:
            raise ValueError("Output file URL points outside the demo origin.")
        with urlopen(url, timeout=30) as response:
            return response.read()

    for path in fixtures[:1 if args.smoke else 3]:
        restore(path)

    records = []
    with ThreadPoolExecutor(max_workers=7) as pool:
        for path in samples:
            start = time.perf_counter()
            try:
                *images, diagnostics = restore(path)
                ready = time.perf_counter() - start
                if len(images) != 7 or "Server processing time" not in diagnostics:
                    raise ValueError("Expected three outputs, four maps and measured diagnostics.")
                files = list(pool.map(download, images))
                elapsed = time.perf_counter() - start
                sizes = []
                for output in files:
                    with Image.open(BytesIO(output)) as image:
                        sizes.append(image.size)
                        image.verify()
                distinct = len({hashlib.sha256(blob).hexdigest() for blob in files[:3]})
                if len(set(sizes)) != 1 or distinct < 2:
                    raise ValueError("Outputs/maps have incompatible sizes or all restorations match.")
                records.append({"input": path.name, "size": sizes[0], "seconds": elapsed,
                                "output_ready_seconds": ready, "distinct_restorations": distinct, "status": "ok"})
                print(f"{path.name}: {elapsed:.2f}s (outputs ready {ready:.2f}s), 3 outputs and 4 maps", flush=True)
            except Exception as exc:
                records.append({"input": path.name, "seconds": time.perf_counter() - start,
                                "status": "error", "error": str(exc)})
                print(f"{path.name}: {exc}", flush=True)

    times = [r["seconds"] for r in records if r["status"] == "ok"]
    result = {"transport": "Gradio HTTP API with image upload and seven concurrent image downloads", "smoke": args.smoke,
              "successful": len(times), "total": len(samples), "warmups": 1 if args.smoke else 3,
              "median_seconds": float(np.median(times)) if times else None,
              "p95_seconds": float(np.percentile(times, 95)) if times else None,
              "max_seconds": max(times) if times else None,
              "all_under_10_seconds": len(times) == len(samples) and max(times) < 10,
              "records": records}
    out = ROOT / "outputs/submission" / ("hosted_parallel_smoke.json" if args.smoke else "hosted_latency.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(out, flush=True)
    if len(times) != len(samples):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
