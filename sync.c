/*
Author - Soumyadip Ghosh
*/

#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BEGIN 1  /* message tag */
#define LTAG 2   /* message tag */
#define RTAG 10  /* message tag */
#define NONE -1  /* indicates no neighbor */
#define DONE 5   /* message tag */
#define MASTER 0 /* taskid of first process */
#define TOL 1e-8 /* tolerance for convergence */

void readData(double *, double *, int, int, int);
void update(int, int, int, int, int, long int, double *, double *,
            double *, double *);
void calc_residual(int, int, int, int, int, long int, double *, double *,
                   double *, double *);
double calc_residual_sum(int, int, int, int, int, double *);
double calc_residual_max(int, int, int, int, int, double *);
double calc_array_sum(int, int, double *);
double calc_array_max(int, int, double *);

int main(int argc, char *argv[]) {
  int NXPROB = atoi(argv[1]);      // x dimension of domain
  int file_write = atoi(argv[2]);  // file write during solver

  int NYPROB; // y dimension of domain
  int NZPROB; // z dimension of domain

  // These dimensions are chosen for our specific use case.
  // Please change them accordingly for your case.
  if (NXPROB == 1600) {
    NYPROB = 100;
    NZPROB = 100;
  }
  else if (NXPROB == 2400) {
    NYPROB = 150;
    NZPROB = 150;
  } 
  else {
    NYPROB = NXPROB;
    NZPROB = NXPROB;
  }

  double *u;        // solution
  double *mat;      // matrix
  double *b;        // rhs
  double *residual; // residual
  double *e;        // error between consecutive iterations

  int taskid = 0,           // this task's unique id
      numtasks = 0,         // number of tasks
      rows = 0, offset = 0, // for domain decomposition
      dest = 0, source = 0, // for message send-receive
      left = 0, right = 0,  // neighbor tasks
      msgtype = 0,          // for message types
      start = 0, end = 0,   // for sub-domain assigned to this rank
      i = 0, ix = 0, iy = 0, iz = 0, it = 0; // loop variables
  long int wsteps = 0;      // compute iterations
  MPI_Status status;
  MPI_Request req1, req2, req3, req4;
  double tstart = 0.0, tend = 0.0;
  double cpu_time_used = 0.0;
  char name[30], send[30], res[30], grid_str[10], proc_str[4],
       task_str[4];

  // iteration residuals
  double local_avg_residual = 0.0;
  double local_max_residual = 0.0;
  double global_avg_residual = 0.0;
  double global_max_residual = 0.0;

  double init_avg_residual = 0.0;
  double init_max_residual = 0.0;

  // another possible convergence criterion - iteration error
  double local_avg_error = 0.0;
  double local_max_error = 0.0;
  double global_avg_error = 0.0;
  double global_max_error = 0.0;

  // Augmented domain considering boundary conditions on each side
  int dom_dim_x = NXPROB + 2;
  int dom_dim_y = NYPROB + 2;
  int dom_dim_z = NZPROB + 2;

  // Dimensions of matrix read from file
  int mat_dim_x = NXPROB;
  int mat_dim_y = NYPROB;
  int mat_dim_z = NZPROB;

  // allocate variables related to domain
  u = (double *)malloc(dom_dim_x * dom_dim_y * dom_dim_z * sizeof(double));
  residual =
      (double *)malloc(dom_dim_x * dom_dim_y * dom_dim_z * sizeof(double));

  e = (double *)malloc(dom_dim_x * dom_dim_y * dom_dim_z * sizeof(double));

  // allocate matrix and rhs
  mat =
      (double *)malloc(mat_dim_x * mat_dim_y * mat_dim_z * 7 * sizeof(double));
  b = (double *)malloc(mat_dim_x * mat_dim_y * mat_dim_z * sizeof(double));

  // Boundary Norm parameters (can be useful to compare with event code)
  double left_send_norm = 0.0, right_send_norm = 0.0;

  /* First, find out my taskid and how many tasks are running */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  // initialize all domain arrays
  for (ix = 0; ix < dom_dim_x; ix++) {
    for (iy = 0; iy < dom_dim_y; iy++) {
      for (iz = 0; iz < dom_dim_z; iz++) {
        *(u + ix * dom_dim_y * dom_dim_z + iy * dom_dim_z + iz) = 0.0;

        *(residual + ix * dom_dim_y * dom_dim_z + iy * dom_dim_z + iz) = 0.0;

        *(e + ix * dom_dim_y * dom_dim_z + iy * dom_dim_z + iz) = 0.0;
      }
    }
  }

  // initialize matrix and rhs arrays
  for (ix = 0; ix < mat_dim_x; ix++) {
    for (iy = 0; iy < mat_dim_y; iy++) {
      for (iz = 0; iz < mat_dim_z; iz++) {
        *(mat + ix * mat_dim_y * mat_dim_z * 7 + iy * mat_dim_z * 7 + iz * 7 +
          0) = 0.0;
        *(mat + ix * mat_dim_y * mat_dim_z * 7 + iy * mat_dim_z * 7 + iz * 7 +
          1) = 0.0;
        *(mat + ix * mat_dim_y * mat_dim_z * 7 + iy * mat_dim_z * 7 + iz * 7 +
          2) = 0.0;
        *(mat + ix * mat_dim_y * mat_dim_z * 7 + iy * mat_dim_z * 7 + iz * 7 +
          3) = 0.0;
        *(mat + ix * mat_dim_y * mat_dim_z * 7 + iy * mat_dim_z * 7 + iz * 7 +
          4) = 0.0;
        *(mat + ix * mat_dim_y * mat_dim_z * 7 + iy * mat_dim_z * 7 + iz * 7 +
          5) = 0.0;
        *(mat + ix * mat_dim_y * mat_dim_z * 7 + iy * mat_dim_z * 7 + iz * 7 +
          6) = 0.0;

        *(b + ix * mat_dim_y * mat_dim_z + iy * mat_dim_z + iz) = 0.0;
      }
    }
  }

  // Master (taskid=0) coordinates the domain decomposition
  if (taskid == MASTER) {
    printf("Starting mpi_poisson3D with %d processors.\n", numtasks);

    /* Initialize grid */
    printf("X dim: %d, Y dim = %d, Z dim = %d\n", NXPROB, NYPROB, NZPROB);
    printf("Reading matrices and rhs\n");
    readData(mat, b, mat_dim_x, mat_dim_y, mat_dim_z);

    /* Distribute work assuming uniform domain decomposition */
    rows = mat_dim_x / numtasks;
    offset = 1; // considering row = 0 as boundary condition

    // tstart = MPI_Wtime(); // start measuring time

    for (i = 1; i < numtasks; i++) {
      // rows = (i <= extra) ? averow+1 : averow; //consider uniform grid
      // division
      offset = offset + rows;
      /* Tell each worker who its neighbors are, since they must exchange */
      /* data with each other. */
      left = i - 1; // holds for all PEs except Master
      if (i == numtasks - 1)
        right = 0; // periodic BC
      else
        right = i + 1;

      /*  Now send startup information to all processors above rank 1  */
      dest = i;
      MPI_Send(&offset, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send(&left, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send(&right, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send((b + (offset - 1) * mat_dim_y * mat_dim_z),
               rows * mat_dim_y * mat_dim_z, MPI_DOUBLE, dest, BEGIN,
               MPI_COMM_WORLD);
      MPI_Send((mat + (offset - 1) * mat_dim_y * mat_dim_z * 7),
               rows * mat_dim_y * mat_dim_z * 7, MPI_DOUBLE, dest, BEGIN,
               MPI_COMM_WORLD);
    }

    // values for MASTER
    if (numtasks > 1) {
      left = numtasks - 1; // periodic boundary condition
      right = 1;
    } else { // if there is only 1 PE
      left = NONE;
      right = NONE;
    }

    offset = 1;

    tstart = MPI_Wtime();

  } /* End of domain decomposition part of master code */

  if (taskid != MASTER) // only other processors need to receive data
  {
    /* Receive my offset, rows, neighbors and grid partition from master */
    source = MASTER;
    msgtype = BEGIN;
    MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&left, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&right, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv((b + (offset - 1) * mat_dim_y * mat_dim_z),
             rows * mat_dim_y * mat_dim_z, MPI_DOUBLE, source, msgtype,
             MPI_COMM_WORLD, &status);
    MPI_Recv((mat + (offset - 1) * mat_dim_y * mat_dim_z * 7),
             rows * mat_dim_y * mat_dim_z * 7, MPI_DOUBLE, source, msgtype,
             MPI_COMM_WORLD, &status);
  }

  // common code starts here

  start = offset;
  end = offset + rows - 1;

  sprintf(proc_str, "%d", numtasks);
  sprintf(grid_str, "%d", NXPROB);
  sprintf(task_str, "%d", taskid);

  // File to write send parameters
  FILE *fs;
  strcpy(send, "send");
  strcat(send, task_str);
  // strcat(send, grid_str);
  strcat(send, ".txt");

  if (file_write == 1) {
    fs = fopen(send, "w");
  } 

  do {

    // Call update routine for executing solver iterations
    update(start, end, dom_dim_x, dom_dim_y, dom_dim_z, wsteps, mat, u, b, e);

    // Norm of boundaries
    left_send_norm = calc_array_sum(0, dom_dim_y * dom_dim_z,
                                    (u + offset * dom_dim_y * dom_dim_z)) /
                     dom_dim_y * dom_dim_z;
    right_send_norm = 
        calc_array_sum(0, dom_dim_y * dom_dim_z,
                       (u + (offset + rows - 1) * dom_dim_y * dom_dim_z)) /
        dom_dim_y * dom_dim_z;

    // printing boundary values
    if (file_write == 1) {
      fprintf(fs, "%2.12lf %2.12lf\n ", left_send_norm, right_send_norm);
    }

    // Boundary Exchange with two-sided communication
    if (left != NONE) {
      MPI_Issend((u + offset * dom_dim_y * dom_dim_z), dom_dim_y * dom_dim_z,
                 MPI_DOUBLE, left, RTAG * 1, MPI_COMM_WORLD, &req1);
      source = left;
      msgtype = LTAG;
      MPI_Irecv((u + (offset - 1) * dom_dim_y * dom_dim_z),
                dom_dim_y * dom_dim_z, MPI_DOUBLE, source, msgtype * 1,
                MPI_COMM_WORLD, &req3);
    }
    if (right != NONE) {
      MPI_Issend((u + (offset + rows - 1) * dom_dim_y * dom_dim_z),
                 dom_dim_y * dom_dim_z, MPI_DOUBLE, right, LTAG * 1,
                 MPI_COMM_WORLD, &req2);
      source = right;
      msgtype = RTAG;
      MPI_Irecv((u + (offset + rows) * dom_dim_y * dom_dim_z),
                dom_dim_y * dom_dim_z, MPI_DOUBLE, source, msgtype * 1,
                MPI_COMM_WORLD, &req4);
    }

    MPI_Wait(&req1, &status);
    MPI_Wait(&req3, &status);
    MPI_Wait(&req2, &status);
    MPI_Wait(&req4, &status);

    // Calculate residuals / iteration errors
    calc_residual(start, end, dom_dim_x, dom_dim_y, dom_dim_z, wsteps,
                  mat, u, b, residual);

      local_avg_residual =
          calc_residual_sum(start, end, dom_dim_x, dom_dim_y, dom_dim_z,
                            residual) / (rows * dom_dim_y * dom_dim_z);
      local_max_residual =
          calc_residual_max(start, end, dom_dim_x, dom_dim_y, dom_dim_z,
                            residual);

      local_avg_error =
          calc_residual_sum(start, end, dom_dim_x, dom_dim_y, dom_dim_z,
                            e) / (rows * dom_dim_y * dom_dim_z);

      local_max_error =
          calc_residual_max(start, end, dom_dim_x, dom_dim_y, dom_dim_z,
                            e);

      MPI_Allreduce(&local_avg_residual, &global_avg_residual, 1, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
      global_avg_residual = global_avg_residual / numtasks;

      MPI_Allreduce(&local_max_residual, &global_max_residual, 1, MPI_DOUBLE,
                    MPI_MAX, MPI_COMM_WORLD);

      if (wsteps == 0) {
        init_avg_residual = global_avg_residual;
        init_max_residual = global_max_residual;
      }

      /*
      MPI_Allreduce(&local_avg_error, &global_avg_error, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      global_avg_error = global_avg_error / numtasks;

      MPI_Allreduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      */

    if (taskid == 0)
      printf(
          "At iter: %ld, Avg R - %lf, Max R - %lf, Avg E - %lf, Max E - %lf\n",
          wsteps, global_avg_residual, global_max_residual, global_avg_error,
          global_max_error);
    wsteps = wsteps + 1;

    } while (fabs(global_max_residual / init_max_residual) > TOL); // end while
  //} while (fabs(global_avg_error) > TOL);

  // printf("Out of loop for proc %d after %d iterations\n",taskid, wsteps-1);
  // if(taskid == MASTER) fclose(fptr);

  if (file_write == 1) {
    fclose(fs);
  }

  // Send solution and residual values to MASTER
  if (taskid != MASTER) {
    MPI_Send(&offset, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
    MPI_Send((u + offset * dom_dim_y * dom_dim_z), rows * dom_dim_y * dom_dim_z,
             MPI_DOUBLE, MASTER, DONE, MPI_COMM_WORLD);
    MPI_Send((residual + offset * dom_dim_y * dom_dim_z),
             rows * dom_dim_y * dom_dim_z, MPI_DOUBLE, MASTER, DONE,
             MPI_COMM_WORLD);
  }

  // MASTER gets values from remaining processes and writes to files
  if (taskid == MASTER) {
    tend = MPI_Wtime();

    for (i = 1; i < numtasks; i++) {
      source = i;
      msgtype = DONE;
      MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
      MPI_Recv((u + offset * dom_dim_y * dom_dim_z),
               rows * dom_dim_y * dom_dim_z, MPI_DOUBLE, source, msgtype,
               MPI_COMM_WORLD, &status);
      MPI_Recv((residual + offset * dom_dim_y * dom_dim_z),
               rows * dom_dim_y * dom_dim_z, MPI_DOUBLE, source, msgtype,
               MPI_COMM_WORLD, &status);
    }

    printf("No of steps - %ld\n", wsteps);

    // tend = MPI_Wtime();
    cpu_time_used = (tend - tstart);

    printf("Time measured: %2.4f\n", cpu_time_used);

    // Write file with steady state values
    strcpy(name, "syncvalues");
    strcat(name, proc_str);
    strcat(name, grid_str);
    strcat(name, ".dat");

    FILE *fp;
    fp = fopen(name, "w");
    for (iz = 1; iz < dom_dim_z - 1; iz++) {
      for (iy = 1; iy < dom_dim_y - 1; iy++) {
        for (ix = 1; ix < dom_dim_x - 1; ix++) {
          fprintf(fp, "%lf ",
                  *(u + ix * dom_dim_y * dom_dim_z + iy * dom_dim_z + iz));
        }
      }
    }
    fclose(fp);
  }

  MPI_Finalize();
} /*end of main*/

void update(int start, int end, int nx, int ny, int nz,
            long int wsteps, double *mat, double *u, double *b, double *e) {

  int ix, iy, iz;

  // Matrix dimensions are lesser than that of domain due to boundary conditions
  int mx = nx - 2;
  int my = ny - 2;
  int mz = nz - 2;

  double beta = 1.2; // SOR parameter

  for (ix = start; ix <= end; ix++) {
    for (iy = 1; iy < ny - 1; iy++) {
      for (iz = 1; iz < nz - 1; iz++) {

        *(e + ix * ny * nz + iy * nz + iz) =
            *(u + ix * ny * nz + iy * nz + iz); // storing old solution

        // Applying periodic BC along Y and Z axes
        if (iy == 1)
          *(u + ix * ny * nz + (iy - 1) * nz + iz) =
              *(u + ix * ny * nz + (ny - 2) * nz + iz);
        if (iy == ny - 2)
          *(u + ix * ny * nz + (iy + 1) * nz + iz) =
              *(u + ix * ny * nz + 1 * nz + iz);
        if (iz == 1)
          *(u + ix * ny * nz + iy * nz + (iz - 1)) =
              *(u + ix * ny * nz + iy * nz + (nz - 2));
        if (iz == nz - 2)
          *(u + ix * ny * nz + iy * nz + (iz + 1)) =
              *(u + ix * ny * nz + iy * nz + 1);

        *(u + ix * ny * nz + iy * nz + iz) =
            (1.0 - beta) * *(u + ix * ny * nz + iy * nz + iz) +
            beta *
                (*(b + (ix - 1) * my * mz + (iy - 1) * mz + (iz - 1)) -
                 *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 +
                   (iz - 1) * 7 + 0) *
                     *(u + (ix - 1) * ny * nz + iy * nz + iz) -
                 *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 +
                   (iz - 1) * 7 + 1) *
                     *(u + (ix + 1) * ny * nz + iy * nz + iz) -
                 *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 +
                   (iz - 1) * 7 + 2) *
                     *(u + ix * ny * nz + (iy - 1) * nz + iz) -
                 *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 +
                   (iz - 1) * 7 + 3) *
                     *(u + ix * ny * nz + (iy + 1) * nz + iz) -
                 *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 +
                   (iz - 1) * 7 + 4) *
                     *(u + ix * ny * nz + iy * nz + (iz - 1)) -
                 *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 +
                   (iz - 1) * 7 + 5) *
                     *(u + ix * ny * nz + iy * nz + (iz + 1))) /
                *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 +
                  (iz - 1) * 7 + 6);

        // difference between old and new sol
        *(e + ix * ny * nz + iy * nz + iz) =
            (*(e + ix * ny * nz + iy * nz + iz) -
             *(u + ix * ny * nz + iy * nz + iz));
      }
    }
  }
}

void calc_residual(int start, int end, int nx, int ny, int nz,
                   long int wsteps, double *mat, double *u, double *b,
                   double *residual) {

  int ix, iy, iz;

  int mx = nx - 2;
  int my = ny - 2;
  int mz = nz - 2;

  for (ix = start; ix <= end; ix++) {
    for (iy = 1; iy < ny - 1; iy++) {
      for (iz = 1; iz < nz - 1; iz++) {
        *(residual + ix * ny * nz + iy * nz + iz) =
            (*(b + (ix - 1) * my * mz + (iy - 1) * mz + (iz - 1)) -
             *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 + (iz - 1) * 7 +
               0) *
                 *(u + (ix - 1) * ny * nz + iy * nz + iz) -
             *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 + (iz - 1) * 7 +
               1) *
                 *(u + (ix + 1) * ny * nz + iy * nz + iz) -
             *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 + (iz - 1) * 7 +
               2) *
                 *(u + ix * ny * nz + (iy - 1) * nz + iz) -
             *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 + (iz - 1) * 7 +
               3) *
                 *(u + ix * ny * nz + (iy + 1) * nz + iz) -
             *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 + (iz - 1) * 7 +
               4) *
                 *(u + ix * ny * nz + iy * nz + (iz - 1)) -
             *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 + (iz - 1) * 7 +
               5) *
                 *(u + ix * ny * nz + iy * nz + (iz + 1)) -
             *(mat + (ix - 1) * my * mz * 7 + (iy - 1) * mz * 7 + (iz - 1) * 7 +
               6) *
                 *(u + ix * ny * nz + iy * nz + iz));
      }
    }
  }
}

void readData(double *mat, double *b, int mx, int my, int mz) {

  // Please change the path for the file accordingly
  char path[200], dim_str[10];
  sprintf(dim_str, "%d", mx);
  strcpy(path, "/afs/crc.nd.edu/user/s/sghosh2/Public/relpres/data/3D/coeff");
  strcat(path, dim_str);
  strcat(path, ".dat");

  FILE *fc;
  int i = 0, j = 0, k = 0;
  int e = 0;
  int entries = mx * my * mz;
  double val1 = 0.0, val2 = 0.0, val3 = 0.0, val4 = 0.0, val5 = 0.0, val6 = 0.0,
         val7 = 0.0, val8 = 0.0;

  fc = fopen(path, "r");

  for (e = 0; e < entries; e++) {
    fscanf(fc, " %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf\n", &i, &j, &k, &val1,
           &val2, &val3, &val4, &val5, &val6, &val7, &val8);
    *(mat + (i - 1) * my * mz * 7 + (j - 1) * mz * 7 + (k - 1) * 7 + 0) = val1;
    *(mat + (i - 1) * my * mz * 7 + (j - 1) * mz * 7 + (k - 1) * 7 + 1) = val2;
    *(mat + (i - 1) * my * mz * 7 + (j - 1) * mz * 7 + (k - 1) * 7 + 2) = val3;
    *(mat + (i - 1) * my * mz * 7 + (j - 1) * mz * 7 + (k - 1) * 7 + 3) = val4;
    *(mat + (i - 1) * my * mz * 7 + (j - 1) * mz * 7 + (k - 1) * 7 + 4) = val5;
    *(mat + (i - 1) * my * mz * 7 + (j - 1) * mz * 7 + (k - 1) * 7 + 5) = val6;
    *(mat + (i - 1) * my * mz * 7 + (j - 1) * mz * 7 + (k - 1) * 7 + 6) = val7;

    *(b + (i - 1) * my * mz + (j - 1) * mz + (k - 1)) = val8;
  }
  fclose(fc);
}

double calc_residual_sum(int start, int end, int nx, int ny, int nz,
                         double *obj) {
  int ix, iy, iz;
  double res_sum = 0.0;
  for (ix = start; ix <= end; ix++) {
    for (iy = 1; iy < ny - 1; iy++) {
      for (iz = 1; iz < nz - 1; iz++) {
        res_sum = res_sum + fabs(*(obj + ix * ny * nz + iy * nz + iz)); // sum
      }
    }
  }
  return res_sum;
}

double calc_residual_max(int start, int end, int nx, int ny, int nz,
                         double *obj) {
  int ix, iy, iz;
  double res_max = 0.0;
  for (ix = start; ix <= end; ix++) {
    for (iy = 1; iy < ny - 1; iy++) {
      for (iz = 1; iz < nz - 1; iz++) {
        if (fabs(*(obj + ix * ny * nz + iy * nz + iz)) > res_max)
          res_max = fabs(*(obj + ix * ny * nz + iy * nz + iz));
      }
    }
  }
  return res_max;
}

double calc_array_sum(int start, int end, double *arr) {
  int ix;
  double arr_sum = 0.0;
  for (ix = start; ix < end; ix++) {
    arr_sum = arr_sum + fabs(*(arr + ix));
  }
  return arr_sum;
}

double calc_array_max(int start, int end, double *arr) {
  int ix;
  double arr_max = 0.0;
  for (ix = start; ix < end; ix++) {
    if (fabs(*(arr + ix)) > fabs(arr_max))
      arr_max = fabs(*(arr + ix));
  }
  return arr_max;
}
