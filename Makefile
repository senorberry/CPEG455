CC=gcc
NC=nvcc
CFLAGS=-Wall -g
simple:
	$(CC) $(CFLAGS) -o simple bmp.c
cache:
	$(CC) $(CFLAGS) -o cache bmpCache.c
gpu:
	$(NC) -g -o cudarun bmp.cu
clean:
	rm -rf simple *.bmp cache cudarun
