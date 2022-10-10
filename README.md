# Event-Triggered Communication in Parallel PDE solvers

This repository proposes a novel event-triggered communication algorithm in a parallel SOR solver for the pressure Poisson PDE for multiphase flows. 

## Data

The sample coefficient files can be downloaded from [here](https://drive.google.com/drive/folders/1yLN4ZdNs9yilXHv7kZ5snrx4pdx2VCx9?usp=sharing). The coeff1600.dat file has the coefficients for the 8 x 0.5 x 0.5 domain considered in the paper with a 1600 x 100 x 100 discretization. The simulation results reported in the paper are based on this file. The coeff800.dat file has the coefficients for the same domain but with a different discretization of 800 x 50 x 50. Note that the path to the file has to be changed and the first dimension has to be provided as the first runtime argument accordingly.

## Instructions

MPI (Message Passing Interface) is a dependency. Code for the three types of algorithms described in the paper below can be run as follows:

### Synchronous 

The baseline bulk synchronous parallel solver is implemented in `sync.c`. The runtime args are length of x dimension and a boolean for enabling log writing respectively. One example command is:
`mpirun -np 200 ./sync 1600 0`

### Asynchronous

The asynchronous parallel solver is a special case of the event-triggered communication solver implemented in `event.c`. The runtime args for the event-triggered solver are length of x dimension, horizon for event threshold, decay for event threshold, averaging filter length at sender, averaging filter length at receiver and a boolean for enabling log writing respectively. If a horizon or decay of 0 is selected, it leads to a threshold of 0 and thus it yields the asynchronous solver. Note that the averaging filter lengths at sender and receiver are redundant here. An example run command is:
`mpirun -np 200 ./event 1600 0 0 20 20 0`

### Event-triggered

The event-triggered communication solver is the main focus of this repository and is implemented in `event.c`. As stated before, the runtime args here are length of x dimension, horizon for event threshold, decay for event threshold, averaging filter length at sender, averaging filter length at receiver and
a boolean for enabling log writing respectively. An example run command is:
`mpirun -np 200 ./event 1600 200 0.5 20 20 0`
