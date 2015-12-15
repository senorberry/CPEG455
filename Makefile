CC=gcc
CFLAGS=-Wall -g
simple:
	$(CC) $(CFLAGS) -o simple bmp.c
cache:
	$(CC) $(CFLAGS) -o cache bmpCache.c
clean:
	rm -rf simple *.bmp cache
