# Web Fix Notes

## Why VF3 was previously predicted near 1.1B VND
The first web form allowed the user to manually combine attributes from unrelated vehicles. In the screenshot, VF3 was paired with SUV body type, 5 seats, 4WD and a 2.0 L engine. Those values are not the VF3 configuration represented in the dataset. The model therefore received an out-of-distribution feature combination.

There was a second data-quality issue: a small number of electric listings contained malformed engine strings that parsed into engine displacement values. Electric cars now always use a missing engine-displacement value.

## What changed
1. The form filters all selectable technical attributes by Brand + Model + Year and then by the selections already made.
2. Constant attributes for a selected model/year are shown but disabled.
3. Electric cars show engine size as N/A.
4. One extreme price anomaly is retained in the repaired audit dataset but excluded from model training by a conservative brand-model-year rule.
5. A train-only market reference is shown next to the model prediction.
6. A wide plausibility guardrail catches only clearly implausible web outputs and keeps the raw model prediction visible in the technical details section.

## VF3 regression test
For `VinFast / VF3 Plus / 2025`, the app can only use the observed configuration in the dataset: Hatchback, automatic, electric, 4 seats, rear-wheel drive, and no engine displacement. The current model predicts about **242.9M VND** for a 15,000 km example, while the train-only model-year median is about **239.5M VND**.
