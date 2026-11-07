### Observed failure cases

These cases come from the 100-pair LOL-v2 Real run. Each inspection grid uses the same 128×128 crop for the low-light input, reference, classical result, deep result and hybrid result. The case IDs and crop coordinates let a reviewer regenerate the evidence.

| Pair | What the outputs show | Likely cause and limitation |
|---|---|---|
| `00703` · lowest hybrid PSNR, 14.39 dB | The window frame and chair become visible, but the wall remains dark and grainy compared with the brighter reference. Deep-only reaches 16.33 dB and shows less grain. | Classical exposure gain amplifies residual shadow noise. Blending that candidate with deep restoration carries some noise into the hybrid. |
| `00707` · largest hybrid PSNR drop against deep | The hybrid leaves a mottled texture across the smooth blue panel and tones it darker than the reference. Deep-only scored 29.20 dB, while hybrid scored 19.95 dB. | The classical candidate contributes high-frequency texture in a region where the deep candidate is already smoother. The noise map is a residual estimate and can confuse texture with noise. |
| `00759` · most newly near-clipped hybrid pixels, 0.142% | The classical result makes the window and net harsh and grainy. The hybrid still mutes the table and wall colours relative to the reference. It scores 18.94 dB versus 16.82 dB for deep-only on this pair. | Shadow and highlight controls protect some bright regions, but mixing three rendered RGB outputs can change colour and retain local artifacts. A PSNR gain can coexist with these visible errors. |

For judging, compare the deep and hybrid outputs on the same image before choosing an export. Very dark inputs can still contain unresolved texture and colour uncertainty.
