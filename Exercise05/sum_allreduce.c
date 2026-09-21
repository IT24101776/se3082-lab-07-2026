#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) fprintf(stderr, "N must be evenly divisible by the number of processes.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int chunk_size = N / size;
    int *array = NULL;
    int *local_chunk = malloc((size_t)chunk_size * sizeof(*local_chunk));
    if (rank == 0) array = malloc((size_t)N * sizeof(*array));
    if (local_chunk == NULL || (rank == 0 && array == NULL)) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    if (rank == 0)
        for (int i = 0; i < N; i++) array[i] = i + 1;

    double start = MPI_Wtime();
    MPI_Scatter(array, chunk_size, MPI_INT, local_chunk, chunk_size,
                MPI_INT, 0, MPI_COMM_WORLD);
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++) local_sum += local_chunk[i];

    long long total_sum = 0;
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("Rank %d: local_sum = %lld, total_sum = %lld, contribution = %.2f%%\n",
           rank, local_sum, total_sum, 100.0 * local_sum / total_sum);

    if (rank == 0) {
        long long expected = (long long)N * (N + 1) / 2;
        printf("[Allreduce] Expected = %lld\n", expected);
        printf("[Allreduce] Correct? = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Allreduce] Time = %.4f sec\n", MPI_Wtime() - start);
    }

    free(array);
    free(local_chunk);
    MPI_Finalize();
    return EXIT_SUCCESS;
}