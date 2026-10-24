Endpoint and margin sensitivity of HDC versus gradient adaptation

The original stream endpoint includes both held-in and held-out sessions. The held-out endpoint and margin sensitivity below are post hoc analyses of the same saved outer-stream predictions; no model is selected or refitted. Paired tests use ten seeds and are conditional on the fixed benchmark sessions.

| Endpoint | Margin | Unadjusted | Holm, 12 tests | Bonferroni, 12 tests |
|:--|--:|--:|--:|--:|
| stream_acc | 0.01 | 9/12 | 7/12 | 7/12 |
| stream_acc | 0.02 | 11/12 | 11/12 | 10/12 |
| stream_acc | 0.03 | 11/12 | 11/12 | 11/12 |
| stream_acc | 0.05 | 11/12 | 11/12 | 11/12 |
| held_out_r2 | 0.01 | 5/12 | 5/12 | 4/12 |
| held_out_r2 | 0.02 | 7/12 | 6/12 | 6/12 |
| held_out_r2 | 0.03 | 8/12 | 8/12 | 8/12 |
| held_out_r2 | 0.05 | 9/12 | 9/12 | 9/12 |

Held-out endpoint at margin 0.03

| Task | Reference | Difference | 90% seed interval | Bonferroni p | 95% crossed-bootstrap interval |
|:--|:--|--:|:--|--:|:--|
| H1 | FT-full | -0.0370 | (-0.0416, -0.0324) | 1 | (-0.0587, -0.0140) |
| H1 | Replay | -0.1031 | (-0.1177, -0.0884) | 1 | (-0.1404, -0.0655) |
| H1 | FT-full-tag | -0.0093 | (-0.0184, -0.0001) | 0.01477 | (-0.0476, +0.0280) |
| H1 | FT-readout-tag | +0.0016 | (-0.0048, +0.0081) | 4.96e-05 | (-0.0196, +0.0267) |
| M1 | FT-full | +0.0433 | (+0.0374, +0.0492) | 1.649e-08 | (+0.0052, +0.1006) |
| M1 | Replay | +0.0065 | (-0.0004, +0.0134) | 2.805e-05 | (-0.0082, +0.0230) |
| M1 | FT-full-tag | -0.0180 | (-0.0226, -0.0133) | 0.006374 | (-0.0444, +0.0034) |
| M1 | FT-readout-tag | -0.0064 | (-0.0104, -0.0023) | 1.193e-05 | (-0.0126, -0.0005) |
| M2 | FT-full | -0.0679 | (-0.0754, -0.0604) | 1 | (-0.0988, -0.0432) |
| M2 | Replay | -0.0594 | (-0.0665, -0.0523) | 1 | (-0.0930, -0.0292) |
| M2 | FT-full-tag | +0.0517 | (+0.0413, +0.0620) | 9.394e-07 | (-0.0134, +0.1210) |
| M2 | FT-readout-tag | +0.0546 | (+0.0483, +0.0610) | 9.405e-09 | (+0.0251, +0.0855) |

The crossed bootstrap resamples seeds and sessions independently, retaining pairing between methods and separate held-in/held-out strata for stream accuracy. Its intervals describe sensitivity to the observed sessions. Only 7, 3, and 4 held-out sessions are available for H1, M1, and M2; seeds are repeated training runs, not additional participants or recording sessions. The formal non-inferiority decisions above use the seed-level test, not these descriptive bootstrap intervals.
