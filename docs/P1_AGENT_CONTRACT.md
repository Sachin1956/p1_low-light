# P1 implementation contract

Applies to Gemini (including the model the team calls Gemini 3.8 Flash), other coding agents and human edits. The model label is user-supplied, not a verified model release. Read `P1_ML_EXECUTION_PLAN.md` before implementing, training, benchmarking, changing metrics or preparing submission claims.

Branch assignment is explicit: project owner and their agents own Part A classical CV, Part C integration and the shared evaluator; teammate and their agents own Part B deep learning on Kaggle, then the app/deployment after the H+4 handoff. Record actual names at H0. Do not modify another owner's branch without an agreed handoff. Each branch exports the interface, five shared images and provenance specified in the plan. Model agents on Agy/Claude/Gemini must report branch status separately; completion of Part A never implies Part B training is complete.

These instructions reduce shortcut risk; they are not a security boundary. A person or independent review process must verify evidence. An agent with write access to both implementation and checks can falsify both. Run sealed evaluation with references inaccessible to inference, and retain trusted manifests/checks outside the implementing agent's writable area where practical.

## Hard rules

1. **Real inference:** load verified compatible weights. Missing weights, dependency, dataset or GPU is a visible error or BLOCKED state. Never silently substitute identity, gamma, random weights, a cached example or another network for the named model. A user-visible classical fallback is permitted only when explicitly labelled and excluded from deep/hybrid benchmark certification.
2. **Reference isolation:** inference inputs are low image, approved config and checkpoint only. No GT mean/exposure scaling, reference histogram/white balance matching, reference-guided crop/alignment or target-derived fusion weights. References are available only to training losses and the separate post-inference scorer. Audit upstream test scripts before reuse.
3. **Split integrity:** hash/freeze manifests before selection. Train on train; select on validation; report on sealed test. Group related scenes and inspect duplicates. Disclose upstream checkpoint exposure. Test filenames/IDs cannot route to hand-picked parameters, checkpoints or saved answers.
4. **Real measurements:** results originate from saved current-run predictions and actual timers. No invented, rounded-from-paper, copied, random, interpolated or hardcoded performance numbers. Missing measurement is null with reason or NOT RUN. No proxy labelled PSNR/LPIPS/NIQE. UI placeholders cannot enter the report.
5. **Complete coverage:** report every manifest item, errors and metric validity counts. No silent skip, easiest-subset scoring, cherry-picked aggregate, altered reference, hidden resize or removal of poor examples. Debug subsets are visibly labelled and cannot certify final quality.
6. **Frozen evaluation:** pin RGB/range/window/border/backbone/precision conventions and selection gates. Do not relax thresholds, replace metric implementations or alter fixtures to pass. A justified protocol correction creates a new version and reruns every comparator; retain the old evidence and disclose the correction.
7. **Actual hybrid:** real classical and deep branches feed the selector/fusion. Report genuine weights/intermediates from the same call. Branch ablations must affect the corresponding fixture outputs. A no-op branch or relabelled deep-only output cannot certify hybrid completion.
8. **Training honesty:** certify fine-tuning only with data provenance, optimizer steps, finite gradients, parameter changes, saved weights and validation results. A loss plot from random/synthetic debug data does not establish real adaptation. Record upstream versus team work and all unsuccessful experiments.
9. **Artifact provenance:** hash checkpoints, configs, splits and predictions; capture source revision and uncommitted patch hash. Resume only compatible runs. Cache entries must key on image bytes, checkpoint, config and code version; exclude cache hits from uncached latency measurements.
10. **Fail visibly:** bad inputs, OOM, nonfinite outputs, incompatible state dicts and unavailable metrics produce explicit errors. No broad exception handler that returns a plausible picture/score and says success. Avoid `strict=False` weight loading unless every unmatched key is explained and reviewed.
11. **Evidence over assertions:** tests exercise real model loading, transformations and scorer behavior. Do not weaken/skip tests or mock the core path to obtain a green run. Unit mocks are labelled and cannot stand in for integration evidence. No TODO, `pass`, `NotImplementedError`, swallowed exceptions or dummy returns in release-critical paths; inspect each occurrence rather than banning legitimate syntax everywhere.
12. **Claims and completion:** measured, reproduced, trained, deployed and state of the art each require matching evidence. Record status as PLANNED / IMPLEMENTED-UNVERIFIED / VERIFIED / BLOCKED. Never mark a task VERIFIED from code inspection alone. Do not claim deterministic networks cannot invent details. Follow the elapsed-time cutoffs; unavailable compute does not authorize fabrication.
13. **Kaggle handoff:** a Kaggle notebook is a training environment, not the application or source of benchmark truth. Export the checkpoint, source revision, environment, training/validation manifest hashes, config and five input-only predictions by H+4. Re-run the checkpoint through the common inference interface after export. If the teammate cannot export it, mark Part B BLOCKED and integrate the verified pretrained checkpoint instead.

## Required independent acceptance checks

Implement checks before the final freeze; keep expected inputs and outcomes reviewable. A coding agent must provide the commands and unedited logs, not merely state that checks passed.

| Adversarial check | Passing behavior |
|---|---|
| Rename/remove checkpoint | Model startup fails clearly; no replacement image |
| Wrong/incompatible checkpoint | Compatibility check fails with actionable message |
| References unavailable during inference | Same predictions as normal input-only run; scorer alone later opens references |
| Rename same input bytes / move path | Same output within declared numerical tolerance; no filename routing |
| Change one input / model config | Corresponding output and provenance change; stale cached answer rejected |
| Inject NaN into candidate/weights | Explicit failure; no silently certified result |
| Corrupt/omit one predicted file | Evaluation refuses complete certification and identifies missing item |
| Identical prediction/reference fixture | MSE zero, PSNR +inf handled explicitly, SSIM ~1 and LPIPS ~0 within implementation tolerance |
| Add controlled error / swap RGB order | Score changes in expected direction on a suitable colour fixture |
| Black image offered as “denoised” result | Paired fidelity/exposure checks reveal failure; low estimated noise alone cannot pass quality |
| Disable CLAHE / adaptivity / each fusion branch / guard | Applicable fixture or validation output changes; logged ablations remain attributable |
| Metric assets unavailable or image too small | Null/error plus reason, never fabricated valid score |
| Regenerate report | Same table values from raw records; no handwritten metric cells |

A team member independently recomputes PSNR on five saved pairs using a separate small implementation (agreement within 1e-4 dB except explicit infinite case), checks SSIM/LPIPS identical and degraded fixtures, inspects five predictions, and repeats one full inference/evaluation command. Review dependency code for GT adjustment and inspect benchmark-path file accesses. Metric unit fixtures can be synthetic; final benchmark images cannot be replaced by them.

## Handoff format for each milestone

Return: milestone and status; changed paths; exact commands executed and exit status; hardware/config/split/checkpoint hashes; artifact locations and image counts; measured results; failed checks; unresolved blockers; next authorized step. Use NOT RUN for commands not executed. Preserve raw failed runs. Requests for more compute identify the failing operation, measured memory/time and minimal resource need.

## Pasteable task for the implementing model

> Implement your assigned P1 branch following docs/P1_ML_EXECUTION_PLAN.md and this contract. The project owner does Part A classical CV; the teammate does Part B training on Kaggle; a named integrator does Part C. We have 18 hours; the owner's RTX 3050 has 4 GB VRAM. Follow the shared handoff interface and H+4 checkpoint export gate. Start at the first incomplete milestone. Produce real predictions and evidence. Keep inference independent of references. Read official model evaluation code for GT-dependent corrections before reuse. Record executed commands, hashes, predictions and metrics. Apply acceptance checks; report failures and NOT RUN honestly. Complete each definition of done before claiming VERIFIED. Do not change metric conventions, selection thresholds or fixtures to manufacture a pass. Finish evaluation and deployment within the cutoff schedule. Distinguish implementation, measurements and blockers.
