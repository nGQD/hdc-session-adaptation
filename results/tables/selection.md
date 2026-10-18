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
