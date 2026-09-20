#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "N must be evenly divisible by the number of processes.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int *array = malloc((size_t)N * sizeof(*array));
    if (array == NULL) {
        fprintf(stderr, "Rank %d could not allocate the array.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
    }

    double start = MPI_Wtime();
    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = N / size;
    int start_idx = rank * chunk_size;
    long long local_sum = 0;
    for (int i = start_idx; i < start_idx + chunk_size; i++)
        local_sum += array[i];

    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    } else {
        long long total_sum = local_sum;
        for (int source = 1; source < size; source++) {
            long long received_sum;
            MPI_Recv(&received_sum, 1, MPI_LONG_LONG, source, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += received_sum;
        }

        long long expected = (long long)N * (N + 1) / 2;
        printf("[Bcast] Total sum = %lld\n", total_sum);
        printf("[Bcast] Expected = %lld\n", expected);
        printf("[Bcast] Correct? = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Bcast] Time = %.4f sec\n", MPI_Wtime() - start);
    }

    free(array);
    MPI_Finalize();
    return EXIT_SUCCESS;
}