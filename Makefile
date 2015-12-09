CC=gcc
CFLAGS=-Wall -g
simple:
	$(CC) $(CFLAGS) -o simple bmp.c

clean:
	rm simple *.bmp
