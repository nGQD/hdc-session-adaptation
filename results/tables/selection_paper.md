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
