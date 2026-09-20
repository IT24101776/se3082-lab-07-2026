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
    long long *all_sums = rank == 0 ? malloc((size_t)size * sizeof(*all_sums)) : NULL;
    if (rank == 0) array = malloc((size_t)N * sizeof(*array));
    if (local_chunk == NULL || (rank == 0 && (array == NULL || all_sums == NULL)))
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

    if (rank == 0)
        for (int i = 0; i < N; i++) array[i] = i + 1;

    double start = MPI_Wtime();
    MPI_Scatter(array, chunk_size, MPI_INT, local_chunk, chunk_size,
                MPI_INT, 0, MPI_COMM_WORLD);
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++) local_sum += local_chunk[i];

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, all_sums, 1,
               MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long total_sum = 0;
        for (int i = 0; i < size; i++) total_sum += all_sums[i];
        long long expected = (long long)N * (N + 1) / 2;
        printf("[Gather] Total sum = %lld\n", total_sum);
        printf("[Gather] Expected = %lld\n", expected);
        printf("[Gather] Correct? = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Gather] Time = %.4f sec\n", MPI_Wtime() - start);
    }

    free(array);
    free(local_chunk);
    free(all_sums);
    MPI_Finalize();
    return EXIT_SUCCESS;
}