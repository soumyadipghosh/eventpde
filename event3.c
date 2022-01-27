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
#define EXTRA_ITER 20 //10

void readData(double *, double *, int, int, int);
void update(int, int, int, int, int, long int, double *, double *, double *,
            double *);
void calc_residual(int, int, int, int, int, long int, double *, double *,
                   double *, double *);
double calc_residual_sum(int, int, int, int, int, double *);
double calc_residual_max(int, int, int, int, int, double *);
double calc_array_sum(int, int, double *);
double calc_array_max(int, int, double *);

int main(int argc, char *argv[]) {
  int NXPROB = atoi(argv[1]);       // x dimension of domain
  int horizon = atoi(argv[2]);      // horizon parameter for event
  double decay = atof(argv[3]);     // decay parameter for event
  int send_history = atoi(argv[4]); // history of sender slopes
  int recv_history = atoi(argv[5]); // history of receiver slopes
  int file_write = atoi(argv[6]);   // file write during solver

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
  long int wpasses = 0;     // passes = compute iterations + idle iterations
  MPI_Status status;
  double tstart = 0.0, tend = 0.0;
  double cpu_time_used = 0.0;
  char name[30], send[30], recv[30], res[30], grid_str[10], task_str[4],
      proc_str[4];

  // Window for MPI one-sided communication
  double *win_mem;
  MPI_Win win;

  // convergence criterion - local residuals
  double init_local_avg_residual = 0.0;
  double init_local_max_residual = 0.0;
  double local_avg_residual = 0.0;
  double local_max_residual = 0.0;

  // global residuals for just the initial and final iteration
  double init_global_avg_residual = 0.0;
  double init_global_max_residual = 0.0;
  double final_global_avg_residual = 0.0;
  double final_global_max_residual = 0.0;

  // another possible convergence criterion - iteration error
  double local_avg_error = 0.0;
  double local_max_error = 0.0;

  // Variables for monitoring distributed convergence
  double localcv = 0.0, globalcv = 0.0;
  double slocalcv = 0.0;
  double tempcv = 1.0;
  double prev_localcv = 0.0;

  // Current and previously received values at boundaries
  double left_recv = 0.0, prev_left_recv = 0.0;
  double right_recv = 0.0, prev_right_recv = 0.0;

  // Counter for extra iterations
  int extra_steps = 0;

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

  // EVENT PARAMETERS FROM HERE
  // Sender Event parameters
  double left_send_norm = 0.0, right_send_norm = 0.0;
  double prev_left_send_norm = 0.0, prev_right_send_norm = 0.0;
  double *left_send_slopes, *right_send_slopes;
  int prev_left_send_iter = 0, prev_right_send_iter = 0;
  double left_send_diff = 0.0, right_send_diff = 0.0;
  int left_send_iter_diff = 0, right_send_iter_diff = 0;

  // Receiver Event parameters
  double left_recv_norm = 0.0, right_recv_norm = 0.0;
  double prev_left_recv_norm = 0.0, prev_right_recv_norm = 0.0;
  double *prev_left_recv_bdy, *prev_right_recv_bdy;
  double *left_recv_slopes, *right_recv_slopes;
  int prev_left_recv_iter = 0, prev_right_recv_iter = 0;
  double left_recv_diff = 0.0, right_recv_diff = 0.0;
  int left_recv_iter_diff = 0, right_recv_iter_diff = 0;

  // Other parameters
  double left_thres = 0.0, right_thres = 0.0;
  int left_msg = 0, right_msg = 0;

  left_send_slopes = (double *)malloc(send_history * sizeof(double));
  right_send_slopes = (double *)malloc(send_history * sizeof(double));

  prev_left_recv_bdy = (double *)malloc(dom_dim_y * dom_dim_z * sizeof(double));
  prev_right_recv_bdy =
      (double *)malloc(dom_dim_y * dom_dim_z * sizeof(double));

  left_recv_slopes =
      (double *)malloc(recv_history * dom_dim_y * dom_dim_z * sizeof(double));
  right_recv_slopes =
      (double *)malloc(recv_history * dom_dim_y * dom_dim_z * sizeof(double));

  for (it = 0; it < send_history; it++) {
    *(left_send_slopes + it) = 0.0;
    *(right_send_slopes + it) = 0.0;
  }

  for (iy = 0; iy < dom_dim_y; iy++) {
    for (iz = 0; iz < dom_dim_z; iz++) {
      *(prev_left_recv_bdy + iy * dom_dim_z + iz) = 0.0;
      *(prev_right_recv_bdy + iy * dom_dim_z + iz) = 0.0;

      for (it = 0; it < recv_history; it++) {
        *(left_recv_slopes + it * dom_dim_y * dom_dim_z + iy * dom_dim_z + iz) =
            0.0;
        *(right_recv_slopes + it * dom_dim_y * dom_dim_z + iy * dom_dim_z +
          iz) = 0.0;
      }
    }
  }
  // END EVENT PARAMETERS

  /* First, find out my taskid and how many tasks are running */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  // create memory window for MPI RMA
  // TODO - RMA window structure
  win_mem = (double *)calloc(2 * dom_dim_y * dom_dim_z + 1 + numtasks,
                             sizeof(double));
  MPI_Win_create(win_mem,
                 (2 * dom_dim_y * dom_dim_z + 1 + numtasks) * sizeof(double),
                 sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

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

  // initialize RMA window
  for (i = 0; i < (2 * dom_dim_y * dom_dim_z + 1 + numtasks); i++) {
    *(win_mem + i) = 0.0;
  }

  // Master (taskid=0) coordinates the domain decomposition
  if (taskid == MASTER) {
    printf("Starting mpi_poisson3D with %d processors.\n", numtasks);

    /* Initialize grid */
    printf("X dim: %d, Y dim: %d, Z dim: %d\n", NXPROB, NYPROB, NZPROB);
    printf("Reading matrices and rhs\n");
    readData(mat, b, mat_dim_x, mat_dim_y, mat_dim_z);

    /* Distribute work assuming uniform domain decomposition */
    rows = mat_dim_x / numtasks;
    offset = 1;

    // tstart = MPI_Wtime(); // start measuring time

    for (i = 1; i < numtasks; i++) {
      offset = offset + rows;
      /* Assign neighbors to tasks */
      left = i - 1;
      if (i == numtasks - 1)
        right = 0; // periodic boundary condition
      else
        right = i + 1;

      /*  Now send startup information to all processors from rank 0  */
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
    /* Receive my offset, rows, neighbors and partitions from master */
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

  // File to write send and recv event parameters
  FILE *fs;
  strcpy(send, "send");
  strcat(send, task_str);
  // strcat(send, grid_str);
  strcat(send, ".txt");

  FILE *fr;
  strcpy(recv, "recv");
  strcat(recv, task_str);
  strcat(recv, ".txt");

  if (file_write == 1) {
    fs = fopen(send, "w");
    fr = fopen(recv, "w");
  }

  do {

    // Local convergence information
    tempcv = 1.0;
    prev_localcv = localcv;

    // Call update routine for executing solver iterations
    update(start, end, dom_dim_x, dom_dim_y, dom_dim_z, wsteps, mat, u, b, e);

    // If local convergence not detected
    if (localcv != 1.0) {

      // Norm of boundaries
      left_send_norm = calc_array_sum(0, dom_dim_y * dom_dim_z,
                                      (u + offset * dom_dim_y * dom_dim_z)) /
                       dom_dim_y * dom_dim_z;
      right_send_norm =
          calc_array_sum(0, dom_dim_y * dom_dim_z,
                         (u + (offset + rows - 1) * dom_dim_y * dom_dim_z)) /
          dom_dim_y * dom_dim_z;

      // Value and iteration difference with that of previous event
      left_send_diff = fabs(left_send_norm - prev_left_send_norm);
      right_send_diff = fabs(right_send_norm - prev_right_send_norm);
      left_send_iter_diff = wsteps - prev_left_send_iter;
      right_send_iter_diff = wsteps - prev_right_send_iter;

      // Decay threshold until next event triggers
      left_thres = left_thres * decay;
      right_thres = right_thres * decay;

      if (file_write == 1)
        fprintf(fs, "%2.12lf %2.12lf %2.12lf %2.12lf ", left_send_norm,
                right_send_norm, left_thres, right_thres);

      // If condition for event is triggered at left boundary after first few iterations
      if (left != NONE && (left_send_diff >= left_thres || wsteps < 2000)) {

        // transfer boundary using one-sided communication
        MPI_Win_lock(MPI_LOCK_SHARED, left, 0, win);
        MPI_Put((u + offset * dom_dim_y * dom_dim_z), dom_dim_y * dom_dim_z,
                MPI_DOUBLE, left, dom_dim_y * dom_dim_z, dom_dim_y * dom_dim_z,
                MPI_DOUBLE, win);
        MPI_Win_unlock(left, win);

        double left_avg_slope = 0.0;

        // push prev slopes
        for (it = 0; it < send_history - 1; it++) {
          *(left_send_slopes + it) = *(left_send_slopes + it + 1);
          left_avg_slope += *(left_send_slopes + it);
        }
        // compute new slope
        if (left_send_iter_diff != 0)
          *(left_send_slopes + it) = left_send_diff / left_send_iter_diff;
        left_avg_slope += *(left_send_slopes + it);

        // assign prev values
        prev_left_send_norm = left_send_norm;
        prev_left_send_iter = wsteps;

        // compute new threshold
        left_avg_slope /= send_history;
        left_thres = left_avg_slope * horizon;

        // increment msg counter
        left_msg++;

        if (file_write == 1)
          fprintf(fs, "%d ", 1); // record for event being triggered
      } else {
        if (file_write == 1)
          fprintf(fs, "%d ", 0); // record for no event being triggered
      }

      // If condition for event is triggered at right boundary after first few iterations
      if (right != NONE && (right_send_diff >= right_thres || wsteps < 2000)) {

        // transfer boundary using one-sided communication
        MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);
        MPI_Put((u + (offset + rows - 1) * dom_dim_y * dom_dim_z),
                dom_dim_y * dom_dim_z, MPI_DOUBLE, right, 0,
                dom_dim_y * dom_dim_z, MPI_DOUBLE, win);
        MPI_Win_unlock(right, win);

        double right_avg_slope = 0.0;

        // push prev slopes
        for (it = 0; it < send_history - 1; it++) {
          *(right_send_slopes + it) = *(right_send_slopes + it + 1);
          right_avg_slope += *(right_send_slopes + it);
        }
        // compute new slope
        if (right_send_iter_diff != 0)
          *(right_send_slopes + it) = right_send_diff / right_send_iter_diff;
        right_avg_slope += *(right_send_slopes + it);

        // assign prev values
        prev_right_send_norm = right_send_norm;
        prev_right_send_iter = wsteps;

        // compute new threshold
        right_avg_slope /= send_history;
        right_thres = right_avg_slope * horizon;

        // increment msg counter
        right_msg++;

        if (file_write == 1)
          fprintf(fs, "%d ", 1); // record for event being triggered
      } else {
        if (file_write == 1)
          fprintf(fs, "%d ", 0); // record for no event being triggered
      }

      if (file_write == 1)
        fprintf(fs, "\n");

      // Calculate norm of received values
      left_recv_norm = calc_array_sum(0, dom_dim_y * dom_dim_z, win_mem) /
                       dom_dim_y * dom_dim_z;
      right_recv_norm = calc_array_sum(0, dom_dim_y * dom_dim_z,
                                       (win_mem + dom_dim_y * dom_dim_z)) /
                        dom_dim_y * dom_dim_z;

      // Value and iteration difference with previously received values
      left_recv_diff = fabs(left_recv_norm - prev_left_recv_norm);
      right_recv_diff = fabs(right_recv_norm - prev_right_recv_norm);
      left_recv_iter_diff = wsteps - prev_left_recv_iter;
      right_recv_iter_diff = wsteps - prev_right_recv_iter;

      // If left neighbor executing compute iterations
      if (left != NONE &&
          *(win_mem + 2 * dom_dim_y * dom_dim_z + 1 + left) != 1.0) {

        // if value difference > 0, it means new values received
        if (left_recv_diff > 0) {
          for (iy = 0; iy < dom_dim_y; iy++) {
            for (iz = 0; iz < dom_dim_z; iz++) {
              // copy to left ghost cells
              *(u + (offset - 1) * dom_dim_y * dom_dim_z + iy * dom_dim_z +
                iz) = *(win_mem + iy * dom_dim_y + iz);

              // push old slopes
              for (it = 0; it < recv_history - 1; it++)
                *(left_recv_slopes + it * dom_dim_y * dom_dim_z +
                  iy * dom_dim_z + iz) =
                    *(left_recv_slopes + (it + 1) * dom_dim_y * dom_dim_z +
                      iy * dom_dim_z + iz);

              // new slope of received values
              if (left_recv_iter_diff != 0)
                *(left_recv_slopes + it * dom_dim_y * dom_dim_z +
                  iy * dom_dim_z + iz) =
                    (*(win_mem + iy * dom_dim_z + iz) -
                     *(prev_left_recv_bdy + iy * dom_dim_z + iz)) /
                    left_recv_iter_diff;
              else
                *(left_recv_slopes + it * dom_dim_y * dom_dim_z +
                  iy * dom_dim_z + iz) = 0.0;

              // update prev bdy
              *(prev_left_recv_bdy + iy * dom_dim_z + iz) =
                  *(win_mem + iy * dom_dim_z + iz);
            }
          }

          prev_left_recv_norm = left_recv_norm;
          prev_left_recv_iter = wsteps;

          if (file_write == 1)
            fprintf(fr, "%d ", 1); // record for value received

        } else { // if new values not received, extrapolate based on history

          for (iy = 0; iy < dom_dim_y; iy++) {
            for (iz = 0; iz < dom_dim_z; iz++) {

              // calculate avg slope for extrapolation
              double avg_slope = 0.0;

              for (it = 0; it < recv_history; it++)
                avg_slope += *(left_recv_slopes + it * dom_dim_y * dom_dim_z +
                               iy * dom_dim_z + iz);

              avg_slope /= recv_history;

              // extrapolated value
              *(u + (offset - 1) * dom_dim_y * dom_dim_z + iy * dom_dim_z +
                iz) = *(prev_left_recv_bdy + iy * dom_dim_z + iz) +
                      avg_slope * left_recv_iter_diff;
            }
          }

          // using recv norm variable to store extrapolated value
          left_recv_norm =
              calc_array_sum(0, dom_dim_y * dom_dim_z,
                             (u + (offset - 1) * dom_dim_y * dom_dim_z)) /
              dom_dim_y * dom_dim_z;

          if (file_write == 1)
            fprintf(fr, "%d ", 0); // record for no value received
        }

        if (file_write == 1)
          fprintf(fr, "%2.12lf ", left_recv_norm);
      } // end left!= NONE

      // If right neighbor executing compute iterations
      if (right != NONE &&
          *(win_mem + 2 * dom_dim_y * dom_dim_z + 1 + right) != 1.0) {

        // if value difference > 0, it means new values received
        if (right_recv_diff > 0) {
          for (iy = 0; iy < dom_dim_y; iy++) {
            for (iz = 0; iz < dom_dim_z; iz++) {
              // copy to right ghost cells
              *(u + (offset + rows) * dom_dim_y * dom_dim_z + iy * dom_dim_z +
                iz) =
                  *(win_mem + (dom_dim_y * dom_dim_z + iy * dom_dim_z + iz));

              // push old slopes
              for (it = 0; it < recv_history - 1; it++)
                *(right_recv_slopes + it * dom_dim_y * dom_dim_z +
                  iy * dom_dim_z + iz) =
                    *(right_recv_slopes + (it + 1) * dom_dim_y * dom_dim_z +
                      iy * dom_dim_z + iz);

              // new slope
              if (right_recv_iter_diff != 0)
                *(right_recv_slopes + it * dom_dim_y * dom_dim_z +
                  iy * dom_dim_z + iz) =
                    (*(win_mem +
                       (dom_dim_y * dom_dim_z + iy * dom_dim_z + iz)) -
                     *(prev_right_recv_bdy + iy * dom_dim_z + iz)) /
                    right_recv_iter_diff;
              else
                *(right_recv_slopes + it * dom_dim_y * dom_dim_z +
                  iy * dom_dim_z + iz) = 0.0;

              // update prev bdy
              *(prev_right_recv_bdy + iy * dom_dim_z + iz) =
                  *(win_mem + (dom_dim_y * dom_dim_z + iy * dom_dim_z + iz));
            }
          }

          prev_right_recv_norm = right_recv_norm;
          prev_right_recv_iter = wsteps;

          if (file_write == 1)
            fprintf(fr, "%d ", 1); // record for value received

        } else { // if new values not received, extrapolate based on history
          
          for (iy = 0; iy < dom_dim_y; iy++) {
            for (iz = 0; iz < dom_dim_z; iz++) {

              // calculate avg slope for extrapolation
              double avg_slope = 0.0;

              for (it = 0; it < recv_history; it++)
                avg_slope += *(right_recv_slopes + it * dom_dim_y * dom_dim_z +
                               iy * dom_dim_z + iz);

              avg_slope /= recv_history;

              // extrapolated value
              *(u + (offset + rows) * dom_dim_y * dom_dim_z + iy * dom_dim_z +
                iz) = *(prev_right_recv_bdy + iy * dom_dim_z + iz) +
                      avg_slope * right_recv_iter_diff;
            }
          }

          // using recv norm variable to store extrapolated value
          right_recv_norm =
              calc_array_sum(0, dom_dim_y * dom_dim_z,
                             (u + (offset + rows) * dom_dim_y * dom_dim_z)) /
              dom_dim_y * dom_dim_z;

          if (file_write == 1)
            fprintf(fr, "%d ", 0); // record for no value received
        }

        if (file_write == 1)
          fprintf(fr, "%2.12lf ", right_recv_norm);
      } // end right!= NONE

      if (file_write == 1)
        fprintf(fr, "\n");

      // Calculate local residuals
      calc_residual(start, end, dom_dim_x, dom_dim_y, dom_dim_z, wsteps, mat, u,
                    b, residual);
      local_avg_residual =
          calc_residual_sum(start, end, dom_dim_x, dom_dim_y, dom_dim_z,
                            residual) / (rows * dom_dim_y * dom_dim_z);
      local_max_residual =
          calc_residual_max(start, end, dom_dim_x, dom_dim_y, dom_dim_z,
                            residual);

      // Calculate the norms of global residual only for the first iteration
      // This is used for verification of convergence 
      if (wsteps == 0) {
        init_local_avg_residual = local_avg_residual;
        init_local_max_residual = local_max_residual;

        MPI_Allreduce(&init_local_avg_residual, &init_global_avg_residual, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        init_global_avg_residual =
            init_global_avg_residual / numtasks;

        MPI_Allreduce(&init_local_max_residual, &init_global_max_residual, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      }

      // Calculate norms of iteration error
      local_avg_error =
          calc_residual_sum(start, end, dom_dim_x, dom_dim_y, dom_dim_z, e)
            / (rows * dom_dim_y * dom_dim_z);
      local_max_error =
          calc_residual_max(start, end, dom_dim_x, dom_dim_y, dom_dim_z, e);

      // If supposed local convergence criterion met in this iteration
      if (fabs(local_max_residual / init_local_max_residual) < TOL) {

        slocalcv = 1.0; // Supposed local convergence

        // If supposed local convergence criterion met for few iterations,
        // this is true local convergence
        if (extra_steps == EXTRA_ITER) {
          localcv = 1.0;
          printf("PE %d reached localcv at iter %ld\n", taskid, wsteps);
          extra_steps = 0;
        }
        extra_steps = extra_steps + 1;

      } else {

        // If supposed local convergence criterion not satisfied again,
        // this is not true local convergence but likely the effect of oscillations
        if (slocalcv == 1.0) {
          slocalcv = 0.0;
          extra_steps = 0;
        }
      }

      wsteps = wsteps + 1;
    } // end localcv != 1.0

    // If local convergence detected
    if (localcv == 1.0) {

      if (left != NONE) {

        // Still monitor values received from left neighbor
        left_recv = calc_array_max(0, dom_dim_y * dom_dim_z, win_mem);

        // If values received from left neighbor change significantly, 
        // it means that compute iterations have to be restarted
        if (fabs(left_recv - prev_left_recv) > 1e-3) {
          prev_left_recv = left_recv;
          localcv = 0.0;
          printf("PE %d unconverged at iter %ld, new values from left\n",
                 taskid, wsteps);
        }
      }

      if (right != NONE) {

        // Still monitor values received from right neighbor
        right_recv = calc_array_max(dom_dim_y * dom_dim_z,
                                    2 * dom_dim_y * dom_dim_z, win_mem);

        // If values received from right neighbor change significantly, 
        // it means that compute iterations have to be restarted
        if (fabs(right_recv - prev_right_recv) > 1e-3) {
          prev_right_recv = right_recv;
          localcv = 0.0;
          printf("PE %d unconverged at iter %ld, new values from right\n",
                 taskid, wsteps);
        }
      }

      // If not MASTER, send convergence information to MASTER and neighbors
      if (taskid != MASTER) {

        // This dummy lock and unlock is done just to ensure progress in MPI
        MPI_Win_lock(MPI_LOCK_SHARED, taskid, 0, win);
        MPI_Win_unlock(taskid, win);

        // Only send local convergence information when there is a change
        if (prev_localcv != localcv) {

          // send to MASTER
          MPI_Win_lock(MPI_LOCK_SHARED, MASTER, 0, win);
          MPI_Put(&localcv, 1, MPI_DOUBLE, MASTER,
                  2 * dom_dim_y * dom_dim_z + 1 + taskid, 1, MPI_DOUBLE, win);
          MPI_Win_flush(MASTER, win);
          MPI_Win_unlock(MASTER, win);

          // send to left neighbor
          if (left != NONE && left != MASTER) {
            MPI_Win_lock(MPI_LOCK_SHARED, left, 0, win);
            MPI_Put(&localcv, 1, MPI_DOUBLE, left,
                    2 * dom_dim_y * dom_dim_z + 1 + taskid, 1, MPI_DOUBLE, win);
            MPI_Win_flush(left, win);
            MPI_Win_unlock(left, win);
          }

          // send to right neighbor
          if (right != NONE) {
            MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);
            MPI_Put(&localcv, 1, MPI_DOUBLE, right,
                    2 * dom_dim_y * dom_dim_z + 1 + taskid, 1, MPI_DOUBLE, win);
            MPI_Win_flush(right, win);
            MPI_Win_unlock(right, win);
          }
        }

        globalcv = *(win_mem + 2 * dom_dim_y * dom_dim_z);
      }

      // If MASTER, send local convergence information to only neighbor (right)
      // and check for global convergence
      if (taskid == MASTER) {

        // This dummy lock and unlock is done just to ensure progress in MPI 
        MPI_Win_lock(MPI_LOCK_SHARED, taskid, 0, win);
        MPI_Win_unlock(taskid, win);

        // Only send local convergence information when there is a change 
        if (prev_localcv != localcv) {
       
          // send to right
          MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);
          MPI_Put(&localcv, 1, MPI_DOUBLE, right,
                  2 * dom_dim_y * dom_dim_z + 1 + taskid, 1, MPI_DOUBLE, win);
          MPI_Win_flush(right, win);
          MPI_Win_unlock(right, win);
        }

        // If all processors including master report local convergence,
        // then global convergence detected
        for (i = 1; i < numtasks; i++) {
          tempcv = tempcv && *(win_mem + 2 * dom_dim_y * dom_dim_z + 1 + i);
        }
        globalcv = tempcv && localcv;

        // If global convergence detected, communicate that to other processors
        if (globalcv == 1.0) {

          for (i = 1; i < numtasks; i++) {
            MPI_Win_lock(MPI_LOCK_SHARED, i, 0, win);
            MPI_Put(&globalcv, 1, MPI_DOUBLE, i, 2 * dom_dim_y * dom_dim_z, 1,
                    MPI_DOUBLE, win);
            MPI_Win_flush(i, win);
            MPI_Win_unlock(i, win);
          }

          printf("Global conv detected at steps %ld\n", wsteps);
        }
      }

    } // end localcv == 1

    wpasses = wpasses + 1;

  } while (globalcv != 1.0); // end while

  printf("Out of loop for proc %d after %ld iters and %ld passes, left msg - "
         "%d, right msg - %d\n",
         taskid, wsteps - 1, wpasses - 1, left_msg, right_msg);
  if (file_write == 1) {
    fclose(fs);
    fclose(fr);
  }

  // Calculate the final global residual for verification
  MPI_Allreduce(&local_avg_residual, &final_global_avg_residual, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
  final_global_avg_residual =
      final_global_avg_residual / numtasks;

  MPI_Allreduce(&local_max_residual, &final_global_max_residual, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);

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

    // tend = MPI_Wtime();
    cpu_time_used = (tend - tstart);

    printf("Time measured: %2.4f\n", cpu_time_used);

    // Print ratio of final to init global residuals
    printf("Global avg R ratio - %1.12lf, Global max R ratio - %1.12lf\n",
           final_global_avg_residual / init_global_avg_residual,
           final_global_max_residual / init_global_max_residual);

    // Write file with steady state values
    strcpy(name, "eventvalues");
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

void update(int start, int end, int nx, int ny, int nz, long int wsteps,
            double *mat, double *u, double *b, double *e) {

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

void calc_residual(int start, int end, int nx, int ny, int nz, long int wsteps,
                   double *mat, double *u, double *b, double *residual) {

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
