## budget
| Task | Epochs | Frozen held-in | Frozen held-out | HDC held-out | FT-full ACC |
|:--|:-:|:-:|:-:|:-:|:-:|
| H1 | 25 | 0.643 | 0.325 | 0.340 | 0.470 |
| H1 | 50 | 0.664 | 0.343 | 0.360 | 0.502 |
| H1 | 100 | 0.687 | 0.368 | 0.385 | 0.540 |
| M1 | 25 | 0.786 | 0.608 | 0.677 | 0.714 |
| M1 | 50 | 0.792 | 0.601 | 0.673 | 0.715 |
| M1 | 100 | 0.788 | 0.587 | 0.657 | 0.706 |
| M2 | 25 | 0.518 | 0.350 | 0.336 | 0.416 |
| M2 | 50 | 0.502 | 0.361 | 0.336 | 0.414 |
| M2 | 100 | 0.493 | 0.368 | 0.333 | 0.413 |

## compact
| Method | H1 ACC | M1 ACC | M2 ACC | H1 held-out | M1 held-out | M2 held-out | H1 BWT | M1 BWT | M2 BWT |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Frozen | 0.490 | 0.708 | 0.431 | 0.342 | 0.597 | 0.360 | 0.000 | 0.000 | 0.000 |
| FT-full | 0.501 | 0.712 | 0.414 | 0.393 | 0.622 | 0.405 | −0.018 | −0.010 | −0.042 |
| FT-readout | 0.478 | 0.713 | 0.399 | 0.365 | 0.630 | 0.389 | −0.030 | −0.015 | −0.044 |
| EWC | 0.502 | 0.633 | 0.428 | 0.394 | 0.584 | 0.406 | −0.018 | −0.089 | −0.027 |
| Replay | 0.544 | 0.713 | 0.435 | 0.438 | 0.654 | 0.397 | 0.007 | −0.038 | −0.011 |
| FT-full-tag | 0.502 | 0.745 | 0.389 | 0.365 | 0.683 | 0.277 | 0.000 | 0.000 | 0.000 |
| FT-readout-tag | 0.497 | 0.741 | 0.419 | 0.355 | 0.673 | 0.337 | 0.000 | 0.000 | 0.000 |
| FT-last-tag | 0.494 | 0.722 | 0.402 | 0.350 | 0.628 | 0.303 | 0.000 | 0.000 | 0.000 |
| Closed form | 0.501 | 0.716 | 0.405 | 0.362 | 0.614 | 0.309 | 0.000 | 0.000 | 0.000 |
| Closed form (zero prior) | 0.489 | 0.673 | 0.395 | 0.340 | 0.515 | 0.289 | 0.000 | 0.000 | 0.000 |
| HDC | 0.498 | 0.738 | 0.417 | 0.356 | 0.666 | 0.334 | 0.000 | 0.000 | 0.000 |
| HDC (zero prior) | 0.449 | 0.722 | 0.372 | 0.266 | 0.630 | 0.243 | 0.000 | 0.000 | 0.000 |
| HDC-shared | 0.505 | 0.706 | 0.424 | 0.381 | 0.666 | 0.396 | −0.005 | −0.035 | −0.024 |
| HDC-bound | 0.451 | 0.731 | 0.333 | 0.269 | 0.651 | 0.165 | −0.027 | −0.004 | −0.042 |
| WF zero-shot | 0.216 | 0.547 | 0.220 | 0.148 | 0.453 | 0.181 | 0.000 | 0.000 | 0.000 |
| WF per-session | 0.224 | 0.576 | 0.238 | 0.164 | 0.520 | 0.217 | 0.000 | 0.000 | 0.000 |
| WF sequential | 0.206 | 0.548 | 0.201 | 0.154 | 0.507 | 0.213 | −0.016 | −0.032 | −0.038 |
| RNN zero-shot | 0.502 | 0.698 | 0.448 | 0.378 | 0.585 | 0.351 | 0.000 | 0.000 | 0.000 |
| RNN per-session | 0.507 | 0.705 | 0.452 | 0.388 | 0.601 | 0.358 | 0.000 | 0.000 | 0.000 |
| RNN sequential | 0.471 | 0.690 | 0.430 | 0.375 | 0.627 | 0.385 | −0.044 | −0.043 | −0.041 |

## macs
| Method | H1 MACs | M1 MACs | M2 MACs | H1 plasticity | M1 plasticity | M2 plasticity |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| FT-full | 9.2e11 | 2.4e10 | 9.4e11 | 0.393 | 0.624 | 0.401 |
| FT-readout | 3.6e10 | 7.1e9 | 2.3e10 | 0.371 | 0.637 | 0.373 |
| EWC | 9.7e11 | 3.5e10 | 9.7e11 | 0.394 | 0.600 | 0.402 |
| Replay | 3.7e12 | 4.7e11 | 4.7e11 | 0.431 | 0.682 | 0.388 |
| FT-full-tag | 4.6e11 | 2.4e11 | 2.4e11 | 0.365 | 0.683 | 0.277 |
| FT-readout-tag | 2.0e10 | 7.1e9 | 7.2e9 | 0.355 | 0.673 | 0.337 |
| FT-last-tag | 1.5e10 | 3.7e9 | 9.0e9 | 0.350 | 0.628 | 0.303 |
| Closed form | 4.1e9 | 2.0e9 | 2.2e9 | 0.362 | 0.614 | 0.309 |
| Closed form (zero prior) | 4.1e9 | 2.0e9 | 2.2e9 | 0.340 | 0.515 | 0.289 |
| HDC | 4.0e10 | 1.2e11 | 3.3e10 | 0.356 | 0.666 | 0.334 |
| HDC (zero prior) | 4.0e10 | 1.2e11 | 3.3e10 | 0.266 | 0.630 | 0.243 |
| HDC-shared | 4.0e10 | 1.2e11 | 3.3e10 | 0.378 | 0.663 | 0.388 |
| HDC-bound | 4.0e10 | 1.2e11 | 3.3e10 | 0.314 | 0.660 | 0.238 |

## main

### H1

| Method | ACC (95% CI) | Held-in | Held-out | BWT | Adapt MACs |
|---|---|---|---|---|---|
| Frozen | 0.490 (0.431, 0.551) | 0.663 | 0.342 | 0.000 | 0 |
| FT-full | 0.501 (0.434, 0.569) | 0.628 | 0.393 | −0.018 | 9.2e11 |
| FT-readout | 0.478 (0.413, 0.545) | 0.611 | 0.365 | −0.030 | 3.6e10 |
| EWC | 0.502 (0.435, 0.569) | 0.628 | 0.394 | −0.018 | 9.7e11 |
| Replay | 0.544 (0.477, 0.612) | 0.667 | 0.438 | 0.007 | 3.7e12 |
| FT-full-tag | 0.502 (0.441, 0.559) | 0.663 | 0.365 | 0.000 | 4.6e11 |
| FT-readout-tag | 0.497 (0.444, 0.547) | 0.663 | 0.355 | 0.000 | 2.0e10 |
| FT-last-tag | 0.494 (0.436, 0.555) | 0.663 | 0.350 | 0.000 | 1.5e10 |
| Closed form | 0.501 (0.437, 0.566) | 0.663 | 0.362 | 0.000 | 4.1e9 |
| Closed form (zero prior) | 0.489 (0.424, 0.555) | 0.663 | 0.340 | 0.000 | 4.1e9 |
| HDC | 0.498 (0.436, 0.558) | 0.663 | 0.356 | 0.000 | 4.0e10 |
| HDC (zero prior) | 0.449 (0.394, 0.499) | 0.663 | 0.266 | 0.000 | 4.0e10 |
| HDC-shared | 0.505 (0.443, 0.567) | 0.649 | 0.381 | −0.005 | 4.0e10 |
| HDC-bound | 0.451 (0.391, 0.508) | 0.663 | 0.269 | −0.027 | 4.0e10 |
| WF zero-shot | 0.216 | 0.295 | 0.148 | 0.000 | 0 |
| WF per-session | 0.224 | 0.295 | 0.164 | 0.000 | 1.1e9 |
| WF sequential | 0.206 | 0.267 | 0.154 | −0.016 | 1.1e9 |
| RNN zero-shot | 0.502 | 0.646 | 0.378 | 0.000 | 0 |
| RNN per-session | 0.507 | 0.646 | 0.388 | 0.000 | 7.8e11 |
| RNN sequential | 0.471 | 0.584 | 0.375 | −0.044 | 6.5e10 |

### M1

| Method | ACC (95% CI) | Held-in | Held-out | BWT | Adapt MACs |
|---|---|---|---|---|---|
| Frozen | 0.708 (0.642, 0.769) | 0.792 | 0.597 | 0.000 | 0 |
| FT-full | 0.712 (0.651, 0.766) | 0.779 | 0.622 | −0.010 | 2.4e10 |
| FT-readout | 0.713 (0.654, 0.764) | 0.775 | 0.630 | −0.015 | 7.1e9 |
| EWC | 0.633 (0.589, 0.675) | 0.671 | 0.584 | −0.089 | 3.5e10 |
| Replay | 0.713 (0.674, 0.759) | 0.756 | 0.654 | −0.038 | 4.7e11 |
| FT-full-tag | 0.745 (0.699, 0.802) | 0.792 | 0.683 | 0.000 | 2.4e11 |
| FT-readout-tag | 0.741 (0.704, 0.784) | 0.792 | 0.673 | 0.000 | 7.1e9 |
| FT-last-tag | 0.722 (0.666, 0.779) | 0.792 | 0.628 | 0.000 | 3.7e9 |
| Closed form | 0.716 (0.671, 0.764) | 0.792 | 0.614 | 0.000 | 2.0e9 |
| Closed form (zero prior) | 0.673 (0.613, 0.734) | 0.792 | 0.515 | 0.000 | 2.0e9 |
| HDC | 0.738 (0.700, 0.782) | 0.792 | 0.666 | 0.000 | 1.2e11 |
| HDC (zero prior) | 0.722 (0.691, 0.761) | 0.792 | 0.630 | 0.000 | 1.2e11 |
| HDC-shared | 0.706 (0.663, 0.751) | 0.737 | 0.666 | −0.035 | 1.2e11 |
| HDC-bound | 0.731 (0.694, 0.776) | 0.792 | 0.651 | −0.004 | 1.2e11 |
| WF zero-shot | 0.547 | 0.617 | 0.453 | 0.000 | 0 |
| WF per-session | 0.576 | 0.618 | 0.520 | 0.000 | 3.7e7 |
| WF sequential | 0.548 | 0.578 | 0.507 | −0.032 | 3.7e7 |
| RNN zero-shot | 0.698 | 0.783 | 0.585 | 0.000 | 0 |
| RNN per-session | 0.705 | 0.783 | 0.601 | 0.000 | 1.8e12 |
| RNN sequential | 0.690 | 0.738 | 0.627 | −0.043 | 2.0e10 |

### M2

| Method | ACC (95% CI) | Held-in | Held-out | BWT | Adapt MACs |
|---|---|---|---|---|---|
| Frozen | 0.431 (0.346, 0.496) | 0.501 | 0.360 | 0.000 | 0 |
| FT-full | 0.414 (0.338, 0.476) | 0.423 | 0.405 | −0.042 | 9.4e11 |
| FT-readout | 0.399 (0.317, 0.467) | 0.409 | 0.389 | −0.044 | 2.3e10 |
| EWC | 0.428 (0.350, 0.490) | 0.449 | 0.406 | −0.027 | 9.7e11 |
| Replay | 0.435 (0.362, 0.496) | 0.473 | 0.397 | −0.011 | 4.7e11 |
| FT-full-tag | 0.389 (0.322, 0.462) | 0.501 | 0.277 | 0.000 | 2.4e11 |
| FT-readout-tag | 0.419 (0.343, 0.489) | 0.501 | 0.337 | 0.000 | 7.2e9 |
| FT-last-tag | 0.402 (0.329, 0.475) | 0.501 | 0.303 | 0.000 | 9.0e9 |
| Closed form | 0.405 (0.326, 0.476) | 0.501 | 0.309 | 0.000 | 2.2e9 |
| Closed form (zero prior) | 0.395 (0.316, 0.469) | 0.501 | 0.289 | 0.000 | 2.2e9 |
| HDC | 0.417 (0.345, 0.480) | 0.501 | 0.334 | 0.000 | 3.3e10 |
| HDC (zero prior) | 0.372 (0.317, 0.434) | 0.501 | 0.243 | 0.000 | 3.3e10 |
| HDC-shared | 0.424 (0.341, 0.488) | 0.452 | 0.396 | −0.024 | 3.3e10 |
| HDC-bound | 0.333 (0.260, 0.401) | 0.501 | 0.165 | −0.042 | 3.3e10 |
| WF zero-shot | 0.220 | 0.259 | 0.181 | 0.000 | 0 |
| WF per-session | 0.238 | 0.259 | 0.217 | 0.000 | 2.4e8 |
| WF sequential | 0.201 | 0.189 | 0.213 | −0.038 | 2.4e8 |
| RNN zero-shot | 0.448 | 0.546 | 0.351 | 0.000 | 0 |
| RNN per-session | 0.452 | 0.546 | 0.358 | 0.000 | 4.9e11 |
| RNN sequential | 0.430 | 0.475 | 0.385 | −0.041 | 3.6e10 |

## tests
| Task | HDC vs | ACC diff | Cliff $\delta$ | p | BWT diff | p (BWT) |
|:--|:--|:-:|:-:|:-:|:-:|:-:|
| H1 | Frozen | 0.008 | 0.50 | 0.027 | 0.000 | 1 |
| H1 | FT-full | −0.004 | −0.32 | 0.26 | 0.018 | 0.025 |
| H1 | FT-readout | 0.020 | 0.88 | 0.025 | 0.030 | 0.025 |
| H1 | EWC | −0.004 | −0.32 | 0.26 | 0.018 | 0.025 |
| H1 | Replay | −0.046 | −0.98 | 0.025 | −0.007 | 0.39 |
| H1 | FT-full-tag | −0.005 | −0.20 | 0.26 | 0.000 | 1 |
| H1 | FT-readout-tag | 0.001 | 0.20 | 0.77 | 0.000 | 1 |
| H1 | FT-last-tag | 0.003 | 0.20 | 0.22 | 0.000 | 1 |
| H1 | Closed form | −0.003 | −0.32 | 0.22 | 0.000 | 1 |
| H1 | Closed form (zero prior) | 0.009 | 0.52 | 0.025 | 0.000 | 1 |
| H1 | HDC (zero prior) | 0.049 | 1.00 | 0.025 | 0.000 | 1 |
| H1 | HDC-shared | −0.007 | −0.54 | 0.025 | 0.005 | 0.025 |
| H1 | HDC-bound | 0.047 | 1.00 | 0.025 | 0.027 | 0.025 |
| M1 | Frozen | 0.030 | 1.00 | 0.025 | 0.000 | 1 |
| M1 | FT-full | 0.026 | 1.00 | 0.025 | 0.010 | 0.025 |
| M1 | FT-readout | 0.025 | 1.00 | 0.025 | 0.015 | 0.025 |
| M1 | EWC | 0.105 | 1.00 | 0.025 | 0.089 | 0.025 |
| M1 | Replay | 0.025 | 1.00 | 0.025 | 0.038 | 0.025 |
| M1 | FT-full-tag | −0.007 | −0.72 | 0.025 | 0.000 | 1 |
| M1 | FT-readout-tag | −0.003 | −0.22 | 0.025 | 0.000 | 1 |
| M1 | FT-last-tag | 0.016 | 0.98 | 0.025 | 0.000 | 1 |
| M1 | Closed form | 0.022 | 0.96 | 0.025 | 0.000 | 1 |
| M1 | Closed form (zero prior) | 0.065 | 1.00 | 0.025 | 0.000 | 1 |
| M1 | HDC (zero prior) | 0.016 | 0.96 | 0.025 | 0.000 | 1 |
| M1 | HDC-shared | 0.032 | 1.00 | 0.025 | 0.035 | 0.025 |
| M1 | HDC-bound | 0.007 | 0.56 | 0.025 | 0.004 | 0.025 |
| M2 | Frozen | −0.013 | −0.88 | 0.025 | 0.000 | 1 |
| M2 | FT-full | 0.003 | 0.22 | 0.26 | 0.042 | 0.025 |
| M2 | FT-readout | 0.018 | 0.96 | 0.025 | 0.044 | 0.025 |
| M2 | EWC | −0.010 | −0.82 | 0.025 | 0.027 | 0.025 |
| M2 | Replay | −0.018 | −1.00 | 0.025 | 0.011 | 0.025 |
| M2 | FT-full-tag | 0.028 | 1.00 | 0.025 | 0.000 | 1 |
| M2 | FT-readout-tag | −0.002 | −0.20 | 0.26 | 0.000 | 1 |
| M2 | FT-last-tag | 0.015 | 0.96 | 0.025 | 0.000 | 1 |
| M2 | Closed form | 0.013 | 0.86 | 0.025 | 0.000 | 1 |
| M2 | Closed form (zero prior) | 0.022 | 0.96 | 0.025 | 0.000 | 1 |
| M2 | HDC (zero prior) | 0.046 | 1.00 | 0.025 | 0.000 | 1 |
| M2 | HDC-shared | −0.006 | −0.56 | 0.025 | 0.024 | 0.025 |
| M2 | HDC-bound | 0.084 | 1.00 | 0.025 | 0.042 | 0.025 |

## tost
| Task | Reference | Diff | 90% CI | Non-inferior at 0.03 | p |
|---|---|---|---|---|---|
| H1 | FT-full | −0.004 | (−0.007, −0.000) | yes | 4.9e-08 |
| H1 | Replay | −0.046 | (−0.054, −0.038) | no | 1 |
| H1 | FT-full-tag | −0.005 | (−0.010, 0.001) | yes | 4.5e-06 |
| H1 | FT-readout-tag | 0.001 | (−0.003, 0.004) | yes | 2.7e-08 |
| M1 | FT-full | 0.026 | (0.024, 0.029) | yes | 2.3e-11 |
| M1 | Replay | 0.025 | (0.020, 0.031) | yes | 1.7e-08 |
| M1 | FT-full-tag | −0.007 | (−0.009, −0.005) | yes | 1.7e-09 |
| M1 | FT-readout-tag | −0.003 | (−0.004, −0.002) | yes | 2.2e-11 |
| M2 | FT-full | 0.003 | (−0.000, 0.007) | yes | 1.7e-08 |
| M2 | Replay | −0.018 | (−0.022, −0.014) | yes | 0.00021 |
| M2 | FT-full-tag | 0.028 | (0.023, 0.033) | yes | 3e-09 |
| M2 | FT-readout-tag | −0.002 | (−0.004, 0.001) | yes | 1.1e-09 |

## mixed
| Task | Method | Estimate vs HDC | 95% CI | p |
|---|---|---|---|---|
| H1 | Closed form | 0.003 | (−0.008, 0.014) | 0.57 |
| H1 | Closed form (zero prior) | −0.009 | (−0.020, 0.002) | 0.12 |
| H1 | EWC | 0.004 | (−0.007, 0.015) | 0.48 |
| H1 | FT-full | 0.004 | (−0.007, 0.014) | 0.53 |
| H1 | FT-readout | −0.020 | (−0.030, −0.009) | 0.00043 |
| H1 | Frozen | −0.008 | (−0.019, 0.003) | 0.17 |
| H1 | FT-full-tag | 0.005 | (−0.006, 0.015) | 0.41 |
| H1 | HDC-bound | −0.047 | (−0.058, −0.036) | 2.7e-17 |
| H1 | HDC-shared | 0.007 | (−0.004, 0.018) | 0.2 |
| H1 | HDC (zero prior) | −0.049 | (−0.060, −0.038) | 2.3e-18 |
| H1 | FT-last-tag | −0.003 | (−0.014, 0.007) | 0.54 |
| H1 | FT-readout-tag | −0.001 | (−0.012, 0.010) | 0.91 |
| H1 | Replay | 0.046 | (0.035, 0.057) | 8.1e-17 |
| M1 | Closed form | −0.022 | (−0.033, −0.011) | 6.8e-05 |
| M1 | Closed form (zero prior) | −0.065 | (−0.076, −0.054) | 3e-31 |
| M1 | EWC | −0.105 | (−0.116, −0.094) | 3.1e-78 |
| M1 | FT-full | −0.026 | (−0.037, −0.015) | 2.4e-06 |
| M1 | FT-readout | −0.025 | (−0.036, −0.014) | 5.5e-06 |
| M1 | Frozen | −0.030 | (−0.041, −0.019) | 1.2e-07 |
| M1 | FT-full-tag | 0.007 | (−0.004, 0.018) | 0.21 |
| M1 | HDC-bound | −0.007 | (−0.018, 0.004) | 0.24 |
| M1 | HDC-shared | −0.032 | (−0.043, −0.021) | 1.7e-08 |
| M1 | HDC (zero prior) | −0.016 | (−0.027, −0.005) | 0.005 |
| M1 | FT-last-tag | −0.016 | (−0.027, −0.005) | 0.0033 |
| M1 | FT-readout-tag | 0.003 | (−0.008, 0.014) | 0.59 |
| M1 | Replay | −0.025 | (−0.036, −0.014) | 5.3e-06 |
| M2 | Closed form | −0.013 | (−0.031, 0.006) | 0.19 |
| M2 | Closed form (zero prior) | −0.022 | (−0.041, −0.003) | 0.02 |
| M2 | EWC | 0.010 | (−0.008, 0.029) | 0.28 |
| M2 | FT-full | −0.003 | (−0.022, 0.015) | 0.72 |
| M2 | FT-readout | −0.018 | (−0.037, 0.000) | 0.056 |
| M2 | Frozen | 0.013 | (−0.006, 0.032) | 0.17 |
| M2 | FT-full-tag | −0.028 | (−0.047, −0.009) | 0.0033 |
| M2 | HDC-bound | −0.084 | (−0.103, −0.066) | 1.3e-18 |
| M2 | HDC-shared | 0.006 | (−0.012, 0.025) | 0.51 |
| M2 | HDC (zero prior) | −0.046 | (−0.064, −0.027) | 1.9e-06 |
| M2 | FT-last-tag | −0.015 | (−0.034, 0.003) | 0.11 |
| M2 | FT-readout-tag | 0.002 | (−0.017, 0.020) | 0.87 |
| M2 | Replay | 0.018 | (−0.001, 0.037) | 0.06 |

## pareto
| Task | Non-dominated (point estimate) | P(non-dominated) |
|---|---|---|
| H1 | Frozen, Replay, FT-full-tag, Closed form, HDC-shared | Frozen 1.00, FT-full 0.13, FT-readout 0.00, EWC 0.14, Replay 1.00, FT-full-tag 0.44, FT-readout-tag 0.36, FT-last-tag 0.06, Closed form 0.96, Closed form (zero prior) 0.00, HDC 0.11, HDC (zero prior) 0.00, HDC-shared 0.60, HDC-bound 0.00 |
| M1 | Frozen, FT-full-tag, FT-readout-tag, FT-last-tag, Closed form | Frozen 1.00, FT-full 0.00, FT-readout 0.00, EWC 0.00, Replay 0.00, FT-full-tag 0.72, FT-readout-tag 1.00, FT-last-tag 0.83, Closed form 0.74, Closed form (zero prior) 0.00, HDC 0.00, HDC (zero prior) 0.00, HDC-shared 0.00, HDC-bound 0.00 |
| M2 | Frozen, Replay | Frozen 1.00, FT-full 0.00, FT-readout 0.00, EWC 0.03, Replay 0.64, FT-full-tag 0.00, FT-readout-tag 0.30, FT-last-tag 0.00, Closed form 0.01, Closed form (zero prior) 0.00, HDC 0.01, HDC (zero prior) 0.00, HDC-shared 0.04, HDC-bound 0.00 |

## ablation
| Task | Variant | Method | Delta held-in | Delta held-out (95% CI) | Delta BWT |
|---|---|---|---|---|---|
| H1 | linear | HDC | −0.022 | 0.036 (−0.007, 0.077) | 0.000 |
| H1 | linear | FT-full | −0.037 | 0.036 (−0.001, 0.073) | −0.011 |
| H1 | linear | Frozen | −0.022 | 0.041 (0.004, 0.074) | 0.000 |
| H1 | mlp | HDC | −0.013 | 0.035 (−0.005, 0.077) | 0.000 |
| H1 | mlp | FT-full | −0.040 | 0.029 (−0.006, 0.064) | −0.019 |
| H1 | mlp | Frozen | −0.013 | 0.039 (0.008, 0.069) | 0.000 |
| H1 | standard_gru | HDC | −0.017 | −0.004 (−0.061, 0.043) | 0.000 |
| H1 | standard_gru | FT-full | 0.017 | −0.022 (−0.067, 0.017) | 0.020 |
| H1 | standard_gru | Frozen | −0.017 | 0.005 (−0.035, 0.041) | 0.000 |
| H1 | no_gru | HDC | −0.082 | −0.115 (−0.170, −0.059) | 0.000 |
| H1 | no_gru | FT-full | −0.069 | −0.100 (−0.145, −0.058) | 0.007 |
| H1 | no_gru | Frozen | −0.082 | −0.089 (−0.147, −0.044) | 0.000 |
| H1 | no_online_norm | HDC | −0.006 | −0.056 (−0.142, −0.010) | 0.000 |
| H1 | no_online_norm | FT-full | 0.004 | −0.055 (−0.135, −0.009) | 0.005 |
| H1 | no_online_norm | Frozen | −0.006 | −0.057 (−0.142, −0.011) | 0.000 |
| H1 | no_smoothing | HDC | −0.138 | −0.121 (−0.177, −0.068) | 0.000 |
| H1 | no_smoothing | FT-full | −0.096 | −0.113 (−0.171, −0.063) | 0.019 |
| H1 | no_smoothing | Frozen | −0.138 | −0.113 (−0.163, −0.069) | 0.000 |
| H1 | no_refinement | HDC | 0.004 | −0.015 (−0.048, 0.011) | 0.000 |
| H1 | no_refinement | FT-full | 0.025 | −0.007 (−0.029, 0.015) | 0.015 |
| H1 | no_refinement | Frozen | 0.004 | 0.010 (−0.014, 0.031) | 0.000 |
| M1 | linear | HDC | 0.005 | −0.010 (−0.048, 0.019) | 0.000 |
| M1 | linear | FT-full | 0.006 | −0.005 (−0.041, 0.025) | 0.001 |
| M1 | linear | Frozen | 0.005 | −0.002 (−0.027, 0.025) | 0.000 |
| M1 | mlp | HDC | −0.000 | −0.016 (−0.054, 0.019) | 0.000 |
| M1 | mlp | FT-full | −0.007 | −0.006 (−0.046, 0.024) | −0.006 |
| M1 | mlp | Frozen | −0.000 | −0.004 (−0.042, 0.026) | 0.000 |
| M1 | standard_gru | HDC | −0.012 | −0.024 (−0.041, −0.005) | 0.000 |
| M1 | standard_gru | FT-full | −0.005 | −0.033 (−0.058, −0.013) | 0.005 |
| M1 | standard_gru | Frozen | −0.012 | −0.021 (−0.049, 0.002) | 0.000 |
| M1 | no_gru | HDC | −0.040 | −0.101 (−0.150, −0.039) | 0.000 |
| M1 | no_gru | FT-full | −0.032 | −0.152 (−0.198, −0.103) | 0.005 |
| M1 | no_gru | Frozen | −0.040 | −0.146 (−0.176, −0.114) | 0.000 |
| M1 | no_online_norm | HDC | −0.002 | −0.006 (−0.015, 0.001) | 0.000 |
| M1 | no_online_norm | FT-full | −0.002 | −0.006 (−0.016, 0.001) | 0.001 |
| M1 | no_online_norm | Frozen | −0.002 | −0.011 (−0.024, 0.001) | 0.000 |
| M1 | no_smoothing | HDC | −0.005 | −0.042 (−0.103, 0.003) | 0.000 |
| M1 | no_smoothing | FT-full | −0.003 | −0.037 (−0.090, 0.009) | 0.004 |
| M1 | no_smoothing | Frozen | −0.005 | −0.068 (−0.125, −0.021) | 0.000 |
| M1 | no_refinement | HDC | −0.008 | −0.032 (−0.054, −0.002) | 0.000 |
| M1 | no_refinement | FT-full | −0.002 | −0.019 (−0.043, 0.011) | 0.003 |
| M1 | no_refinement | Frozen | −0.008 | −0.009 (−0.034, 0.027) | 0.000 |
| M2 | linear | HDC | 0.036 | 0.032 (−0.019, 0.095) | 0.000 |
| M2 | linear | FT-full | 0.010 | 0.025 (−0.007, 0.064) | −0.019 |
| M2 | linear | Frozen | 0.036 | 0.005 (−0.038, 0.051) | 0.000 |
| M2 | mlp | HDC | 0.039 | 0.021 (−0.025, 0.082) | 0.000 |
| M2 | mlp | FT-full | 0.003 | 0.005 (−0.027, 0.042) | −0.030 |
| M2 | mlp | Frozen | 0.039 | 0.006 (−0.036, 0.059) | 0.000 |
| M2 | standard_gru | HDC | 0.031 | −0.011 (−0.048, 0.033) | 0.000 |
| M2 | standard_gru | FT-full | 0.052 | 0.016 (−0.038, 0.080) | 0.014 |
| M2 | standard_gru | Frozen | 0.031 | 0.007 (−0.057, 0.076) | 0.000 |
| M2 | no_gru | HDC | −0.157 | −0.109 (−0.169, −0.065) | 0.000 |
| M2 | no_gru | FT-full | −0.122 | −0.170 (−0.240, −0.118) | 0.022 |
| M2 | no_gru | Frozen | −0.157 | −0.154 (−0.218, −0.088) | 0.000 |
| M2 | no_online_norm | HDC | −0.008 | −0.016 (−0.035, −0.001) | 0.000 |
| M2 | no_online_norm | FT-full | −0.005 | −0.030 (−0.062, −0.011) | 0.000 |
| M2 | no_online_norm | Frozen | −0.008 | −0.023 (−0.053, 0.000) | 0.000 |
| M2 | no_smoothing | HDC | −0.043 | −0.049 (−0.118, 0.030) | 0.000 |
| M2 | no_smoothing | FT-full | −0.001 | −0.016 (−0.081, 0.074) | 0.042 |
| M2 | no_smoothing | Frozen | −0.043 | −0.074 (−0.136, 0.011) | 0.000 |
| M2 | no_refinement | HDC | −0.008 | −0.029 (−0.081, 0.013) | 0.000 |
| M2 | no_refinement | FT-full | −0.001 | −0.009 (−0.044, 0.025) | 0.005 |
| M2 | no_refinement | Frozen | −0.008 | 0.000 (−0.045, 0.043) | 0.000 |

## dose
| Task | Method | Held-out R2 at 0.125 | Held-out R2 at 0.25 | Held-out R2 at 0.5 | Adapt MACs at 0.5 |
|---|---|---|---|---|---|
| H1 | HDC | 0.272 | 0.314 | 0.360 | 4.0e10 |
| H1 | Closed form | 0.297 | 0.342 | 0.362 | 4.1e9 |
| H1 | FT-full | 0.341 | 0.369 | 0.393 | 9.2e11 |
| H1 | Replay | 0.345 | 0.395 | 0.445 | 3.7e12 |
| M1 | HDC | 0.620 | 0.649 | 0.673 | 1.2e11 |
| M1 | Closed form | 0.556 | 0.576 | 0.620 | 2.0e9 |
| M1 | FT-full | 0.618 | 0.625 | 0.629 | 2.4e10 |
| M1 | Replay | 0.640 | 0.664 | 0.666 | 4.7e11 |
| M2 | HDC | 0.288 | 0.305 | 0.336 | 3.3e10 |
| M2 | Closed form | 0.187 | 0.284 | 0.315 | 2.2e9 |
| M2 | FT-full | 0.397 | 0.391 | 0.408 | 9.4e11 |
| M2 | Replay | 0.358 | 0.360 | 0.405 | 4.7e11 |

## epochs
| Task | Method | Epochs | n | ACC | BWT | Adapt MACs | MACs / HDC | Smallest E with ACC >= HDC |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| H1 | FT-full | 1 | 5 | 0.494 | 0.001 | 4.6e10 | 1.2x | 5 |
| H1 | FT-full | 2 | 5 | 0.497 | 0.002 | 9.2e10 | 2.3x |  |
| H1 | FT-full | 5 | 5 | 0.502 | 0.002 | 2.3e11 | 5.8x |  |
| H1 | FT-full | 10 | 5 | 0.506 | −0.002 | 4.6e11 | 11.6x |  |
| H1 | FT-full | 20 | 5 | 0.504 | −0.016 | 9.2e11 | 23.1x |  |
| H1 | FT-full | 40 | 5 | 0.485 | −0.048 | 1.8e12 | 46.2x |  |
| H1 | Replay | 1 | 5 | 0.365 | −0.121 | 9.2e10 | 2.3x | 5 |
| H1 | Replay | 2 | 5 | 0.451 | −0.049 | 1.8e11 | 4.6x |  |
| H1 | Replay | 5 | 5 | 0.505 | 0.009 | 4.6e11 | 11.6x |  |
| H1 | Replay | 10 | 5 | 0.511 | 0.002 | 9.2e11 | 23.1x |  |
| H1 | Replay | 20 | 5 | 0.538 | 0.009 | 1.8e12 | 46.2x |  |
| H1 | Replay | 40 | 5 | 0.542 | 0.004 | 3.7e12 | 92.4x |  |
| H1 | FT-readout-tag | 1 | 5 | 0.499 | 0.000 | 4.6e9 | 0.1x | 1 |
| H1 | FT-readout-tag | 2 | 5 | 0.504 | 0.000 | 5.4e9 | 0.1x |  |
| H1 | FT-readout-tag | 5 | 5 | 0.508 | 0.000 | 7.8e9 | 0.2x |  |
| H1 | FT-readout-tag | 10 | 5 | 0.508 | 0.000 | 1.2e10 | 0.3x |  |
| H1 | FT-readout-tag | 20 | 5 | 0.501 | 0.000 | 2.0e10 | 0.5x |  |
| H1 | FT-readout-tag | 40 | 5 | 0.490 | 0.000 | 3.6e10 | 0.9x |  |
| M1 | FT-full | 1 | 5 | 0.714 | −0.004 | 1.2e10 | 0.1x | never |
| M1 | FT-full | 2 | 5 | 0.715 | −0.009 | 2.4e10 | 0.2x |  |
| M1 | FT-full | 5 | 5 | 0.708 | −0.026 | 5.9e10 | 0.5x |  |
| M1 | FT-full | 10 | 5 | 0.697 | −0.044 | 1.2e11 | 1.0x |  |
| M1 | FT-full | 20 | 5 | 0.686 | −0.060 | 2.4e11 | 2.0x |  |
| M1 | FT-full | 40 | 5 | 0.681 | −0.075 | 4.7e11 | 4.0x |  |
| M1 | Replay | 1 | 5 | 0.715 | −0.020 | 2.4e10 | 0.2x | never |
| M1 | Replay | 2 | 5 | 0.714 | −0.022 | 4.7e10 | 0.4x |  |
| M1 | Replay | 5 | 5 | 0.717 | −0.033 | 1.2e11 | 1.0x |  |
| M1 | Replay | 10 | 5 | 0.714 | −0.037 | 2.4e11 | 2.0x |  |
| M1 | Replay | 20 | 5 | 0.719 | −0.032 | 4.7e11 | 4.0x |  |
| M1 | Replay | 40 | 5 | 0.719 | −0.035 | 9.4e11 | 8.0x |  |
| M1 | FT-readout-tag | 1 | 5 | 0.723 | 0.000 | 2.0e9 | 0.0x | 20 |
| M1 | FT-readout-tag | 2 | 5 | 0.728 | 0.000 | 2.1e9 | 0.0x |  |
| M1 | FT-readout-tag | 5 | 5 | 0.732 | 0.000 | 2.5e9 | 0.0x |  |
| M1 | FT-readout-tag | 10 | 5 | 0.738 | 0.000 | 3.2e9 | 0.0x |  |
| M1 | FT-readout-tag | 20 | 5 | 0.742 | 0.000 | 4.5e9 | 0.0x |  |
| M1 | FT-readout-tag | 40 | 5 | 0.743 | 0.000 | 7.1e9 | 0.1x |  |
| M2 | FT-full | 1 | 5 | 0.432 | −0.001 | 2.4e10 | 0.7x | 1 |
| M2 | FT-full | 2 | 5 | 0.433 | −0.002 | 4.7e10 | 1.4x |  |
| M2 | FT-full | 5 | 5 | 0.433 | −0.005 | 1.2e11 | 3.6x |  |
| M2 | FT-full | 10 | 5 | 0.431 | −0.012 | 2.4e11 | 7.2x |  |
| M2 | FT-full | 20 | 5 | 0.425 | −0.025 | 4.7e11 | 14.3x |  |
| M2 | FT-full | 40 | 5 | 0.414 | −0.044 | 9.4e11 | 28.7x |  |
| M2 | Replay | 1 | 5 | 0.415 | −0.032 | 4.7e10 | 1.4x | 2 |
| M2 | Replay | 2 | 5 | 0.419 | −0.033 | 9.4e10 | 2.9x |  |
| M2 | Replay | 5 | 5 | 0.431 | −0.026 | 2.4e11 | 7.2x |  |
| M2 | Replay | 10 | 5 | 0.438 | −0.011 | 4.7e11 | 14.3x |  |
| M2 | Replay | 20 | 5 | 0.445 | −0.010 | 9.4e11 | 28.7x |  |
| M2 | Replay | 40 | 5 | 0.448 | −0.005 | 1.9e12 | 57.3x |  |
| M2 | FT-readout-tag | 1 | 5 | 0.434 | 0.000 | 2.5e9 | 0.1x | 1 |
| M2 | FT-readout-tag | 2 | 5 | 0.434 | 0.000 | 3.0e9 | 0.1x |  |
| M2 | FT-readout-tag | 5 | 5 | 0.427 | 0.000 | 4.6e9 | 0.1x |  |
| M2 | FT-readout-tag | 10 | 5 | 0.419 | 0.000 | 7.2e9 | 0.2x |  |
| M2 | FT-readout-tag | 20 | 5 | 0.407 | 0.000 | 1.2e10 | 0.4x |  |
| M2 | FT-readout-tag | 40 | 5 | 0.387 | 0.000 | 2.3e10 | 0.7x |  |

## selection
Table: Hyperparameters selected on the inner stream, per task and method. lam is the ridge penalty per 1000 bins, dimension the code dimension, scale the FHRR scale, lr the learning rate, epochs the number of adaptation epochs and ewc the EWC penalty weight. The last column lists the axes at which the selected value equals an end of the base grid. {#tab:selection}

| Task | Method | Selected | On grid boundary |
|---|---|---|---|
| H1 | FT-full | lr=1e-05, epochs=20 | lr |
| H1 | FT-readout | lr=3e-05, epochs=40 | epochs |
| H1 | EWC | lr=1e-05, epochs=20, ewc=100.0 | lr, ewc |
| H1 | Replay | lr=0.001, epochs=40 | lr, epochs |
| H1 | FT-full-tag | lr=0.0003, epochs=10 | no |
| H1 | FT-readout-tag | lr=0.001, epochs=20 | lr |
| H1 | FT-last-tag | lr=0.001, epochs=40 | lr, epochs |
| H1 | Closed form | lam=300.0 | no |
| H1 | Closed form (zero prior) | lam=100.0 | no |
| H1 | HDC | source=state, dimension=2048, scale=1.0, lam=3000.0 | source, scale |
| H1 | HDC (zero prior) | lam=1000.0 | lam |
| H1 | HDC-shared | lam=10000.0 | no |
| M1 | FT-full | lr=3e-05, epochs=2 | no |
| M1 | FT-readout | lr=3e-05, epochs=40 | epochs |
| M1 | EWC | lr=0.001, epochs=2, ewc=1000000.0 | lr, ewc |
| M1 | Replay | lr=0.0003, epochs=20 | no |
| M1 | FT-full-tag | lr=0.0001, epochs=20 | no |
| M1 | FT-readout-tag | lr=0.001, epochs=40 | lr, epochs |
| M1 | FT-last-tag | lr=0.001, epochs=40 | lr, epochs |
| M1 | Closed form | lam=30.0 | lam |
| M1 | Closed form (zero prior) | lam=1.0 | no |
| M1 | HDC | source=state, dimension=4096, scale=1.0, lam=1000.0 | source, dimension, scale |
| M1 | HDC (zero prior) | lam=1000.0 | lam |
| M1 | HDC-shared | lam=3000.0 | no |
| M2 | FT-full | lr=1e-05, epochs=40 | lr, epochs |
| M2 | FT-readout | lr=0.0001, epochs=40 | epochs |
| M2 | EWC | lr=1e-05, epochs=40, ewc=1000000.0 | lr, epochs, ewc |
| M2 | Replay | lr=0.0003, epochs=10 | no |
| M2 | FT-full-tag | lr=0.001, epochs=10 | lr |
| M2 | FT-readout-tag | lr=0.001, epochs=10 | lr |
| M2 | FT-last-tag | lr=0.001, epochs=40 | lr, epochs |
| M2 | Closed form | lam=30.0 | lam |
| M2 | Closed form (zero prior) | lam=10.0 | no |
| M2 | HDC | source=state, dimension=2048, scale=1.0, lam=1000.0 | source, scale |
| M2 | HDC (zero prior) | lam=1000.0 | lam |
| M2 | HDC-shared | lam=3000.0 | no |

## selection_paper
Table: Hyperparameters selected on the inner stream, per task and method. Gradient methods: learning rate / epochs (/ EWC penalty weight); closed forms, HDC (zero prior) and HDC-shared: ridge penalty $\lambda$ per 1000 bins; HDC: code dimension $D$ / FHRR scale $\sigma$ / $\lambda$. HDC uses the feature $f_t$ as encoder input on every task, and HDC-bound uses the HDC selection. {#tab:selection}

| Method | H1 | M1 | M2 |
|:--|:--|:--|:--|
| FT-full | 1e-5 / 20 | 3e-5 / 2 | 1e-5 / 40 |
| FT-readout | 3e-5 / 40 | 3e-5 / 40 | 1e-4 / 40 |
| EWC | 1e-5 / 20 / 1e2 | 1e-3 / 2 / 1e6 | 1e-5 / 40 / 1e6 |
| Replay | 1e-3 / 40 | 3e-4 / 20 | 3e-4 / 10 |
| FT-full-tag | 3e-4 / 10 | 1e-4 / 20 | 1e-3 / 10 |
| FT-readout-tag | 1e-3 / 20 | 1e-3 / 40 | 1e-3 / 10 |
| FT-last-tag | 1e-3 / 40 | 1e-3 / 40 | 1e-3 / 40 |
| Closed form | 300 | 30 | 30 |
| Closed form (zero prior) | 100 | 1 | 10 |
| HDC | 2048 / 1 / 3000 | 4096 / 1 / 1000 | 2048 / 1 / 1000 |
| HDC (zero prior) | 1000 | 1000 | 1000 |
| HDC-shared | 10000 | 3000 | 3000 |

## ratios
- H1: FT-full/HDC = 23.1x, FT-readout/HDC = 0.9x, EWC/HDC = 24.3x, Replay/HDC = 92.4x, FT-full-tag/HDC = 11.6x, FT-readout-tag/HDC = 0.5x, FT-last-tag/HDC = 0.4x; HDC/Closed form = 9.7x
- M1: FT-full/HDC = 0.2x, FT-readout/HDC = 0.1x, EWC/HDC = 0.3x, Replay/HDC = 4.0x, FT-full-tag/HDC = 2.0x, FT-readout-tag/HDC = 0.1x, FT-last-tag/HDC = 0.0x; HDC/Closed form = 57.8x
- M2: FT-full/HDC = 28.7x, FT-readout/HDC = 0.7x, EWC/HDC = 29.5x, Replay/HDC = 14.3x, FT-full-tag/HDC = 7.2x, FT-readout-tag/HDC = 0.2x, FT-last-tag/HDC = 0.3x; HDC/Closed form = 15.2x
