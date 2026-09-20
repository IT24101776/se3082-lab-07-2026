CC = mpicc
CFLAGS = -Wall -Wextra -O2
MPIRUN = mpirun
PROCESSES = 4

PROGRAMS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

.PHONY: all run clean

all: $(PROGRAMS)

sum_bcast: Exercise01/sum_bcast.c
	$(CC) $(CFLAGS) -o $@ $<

sum_scatter: Exercise02/sum_scatter.c
	$(CC) $(CFLAGS) -o $@ $<

sum_gather: Exercise03/sum_gather.c
	$(CC) $(CFLAGS) -o $@ $<

sum_reduce: Exercise04/sum_reduce.c
	$(CC) $(CFLAGS) -o $@ $<

sum_allreduce: Exercise05/sum_allreduce.c
	$(CC) $(CFLAGS) -o $@ $<

sum_scan: Exercise06/sum_scan.c
	$(CC) $(CFLAGS) -o $@ $<

run: all
	$(MPIRUN) -np $(PROCESSES) ./sum_bcast
	$(MPIRUN) -np $(PROCESSES) ./sum_scatter
	$(MPIRUN) -np $(PROCESSES) ./sum_gather
	$(MPIRUN) -np $(PROCESSES) ./sum_reduce
	$(MPIRUN) -np $(PROCESSES) ./sum_allreduce
	$(MPIRUN) -np $(PROCESSES) ./sum_scan

clean:
	$(RM) $(PROGRAMS)