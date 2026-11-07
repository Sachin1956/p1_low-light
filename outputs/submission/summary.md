### LOL-v2 Real official test · 100 paired captures

The same app configuration restored every image at native size. Predictions were saved and hashed before downloading the references.

| Method | RGB PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Shadow MAD ↓ | Paired edge correlation ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Classical | 15.9323 | 0.3951 | 0.4944 | 8.9771 | 15.2261 | 0.7414 |
| Deep | 20.0330 | 0.8444 | 0.1559 | 4.4125 | 1.0600 | 0.8765 |
| Hybrid | 21.5660 | 0.7623 | 0.1923 | 6.4309 | 4.3143 | 0.8712 |

Hybrid minus deep PSNR: +1.53 dB. Paired bootstrap 95% interval [+0.87, +2.19] dB, resampled by pair ID because scene grouping is unavailable.

CPU decode + three outputs + map rendering + PNG writes: 1.95 s median, 4.36 s p95, 5.95 s maximum. Browser transfer is timed separately.

This is an audit of the shipped settings. Five test inputs previously appeared in the checkpoint handoff without references. The test split was not used to change the app's inference settings during this audit.
