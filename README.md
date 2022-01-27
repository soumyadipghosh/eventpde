## Event-Triggered Communication in Parallel PDE solvers

This repository implements event-triggered communication in a parallel SOR solver for the Pressure Poisson PDE for multiphase flows. 

# Instructions on code

`sync.c` - Implements the baseline bulk synchronous parallel solver

`event.c` - Implements the event-triggered communication parallel solver. Note that choosing a threshold of 0, i.e., horizon=0 or decay=0, will yield the asynchronous parallel solver.
