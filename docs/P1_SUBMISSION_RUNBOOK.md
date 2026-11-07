# P1 demo and evaluation runbook

## Demo

Install `requirements.txt` and run `python app.py --share` from the repository. Gradio prints the public HTTPS URL. Keep the server process and host connected during judging. A share link is temporary and forwards requests to this machine. A durable deployment needs a managed host running the same entry point. The app loads the exact v3 checkpoint at startup and prints its SHA-256. The v3 training log records zero smartphone pairs, so describe the checkpoint as adapted on LOL-v2 Real.

Upload a low-light photo or select a labelled synthetic preset. Keep the default controls for comparison with the benchmark. The three output panels provide downloads. The map tab shows input illumination, estimated high-frequency noise, edges, and RGB fusion weights before highlight protection. The benchmark tab shows the archived comparison and failure observations. The elapsed time printed in the app covers server processing. Public upload, queue and transfer timings are measured separately.

Images are processed at up to 768 pixels on the long side, with dimensions shown in diagnostics. The service accepts uploads up to 15 MB and 24 megapixels. Gradio caches uploads and downloads temporarily and schedules deletion of cached files older than one hour. Use test images that can be shared with the host.

## Vercel preview

Vercel detects `Dockerfile.vercel` and runs the Gradio server on the assigned `PORT`. The container uses CPU PyTorch and includes the checkpoint, demo fixtures, and evaluation summaries. `.vercelignore` allows only the runtime files. The Vercel deployment caps uploads at 4 MB to leave room for multipart overhead under the platform's 4.5 MB request limit.

From the project root, authenticate with `npx vercel@latest login`, then create a preview with `npx vercel@latest deploy`. Choose the team's Vercel project when prompted. Use the preview URL to run the public upload/download check below before promoting it to production. Check a cold request after the service scales down and review any deployment protection settings before sharing the URL with judges.

## Reproduce the comparative evaluation

Run the commands in this order with the same app and checkpoint. The reference directory must be absent during inference. The source mirror revision and the official test IDs are pinned. Dataset PNGs and the regenerated prediction/crop images stay in ignored directories.

```sh
python tools/download_submission_split.py Input
python tools/submission_benchmark.py infer
python tools/download_submission_split.py GT
python tools/submission_benchmark.py score
```

`outputs/submission/inference.json` records checkpoint, code, manifest and per-image hashes. The scorer verifies all 300 predictions and input hashes before publishing results. `outputs/submission/scores.json` contains per-image results and valid counts. `outputs/submission/summary.md` feeds the app's benchmark tab. Failure cases use identical crops across the input, reference and three outputs. Their generated grids are under `outputs/tmp/submission_lolv2` and their IDs and coordinates are in `outputs/submission/failure_cases.json`.

The recorded run used Python 3.13.14, Gradio 6.31.0, PyTorch 2.15.0 CPU with four threads, OpenCV 5.0.0.93, scikit-image 0.26.0 and LPIPS 0.1.4. LPIPS uses the standard AlexNet ImageNet weights (`7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`) and LPIPS v0.1 Alex calibration (`df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`). NIQE uses the pristine statistics bundled with the upstream HVI code. A separate pixel-by-pixel PSNR calculation agreed with the saved hybrid scores on five pairs to within 0.0001 dB.

## Check the public endpoint

```sh
python tools/audit_hosted_submission.py <printed-public-url>
```

The full audit uses three warmups and 20 evenly spaced test inputs, and downloads all three restored images and four maps over HTTPS. It records request time and output-ready time in `outputs/submission/hosted_latency.json`. Before submission, verify a fresh upload and the download controls from a phone and a laptop. Record cold starts and report any requests above the ten-second target.

The temporary Windows CPU host completed 20 of 20 requests. The observed median was 11.26 s and p95 was 12.80 s including file transfer. The offline scoring job overlapped part of this audit. A separate one-image retest after scoring finished took 13.85 s, so the ten-second target remains unmet. Desktop Chrome displayed the three results and four maps on a synthetic preset. A physical-phone check and three cold-start trials are pending.
