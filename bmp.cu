#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

#define N 1024
#define HEADER_SIZE (54)
#define LENGTH (3*N*N)

#define screenh N
#define screenw N

typedef unsigned char byte_t;

void BMPwrite(byte_t* bmp)
{
  int i;
  FILE *file;
  file = fopen("cuda.bmp", "w+");
  for(i = 0; i < LENGTH; i+=8)
    {
      putc(bmp[i], file);
      putc(bmp[i+1], file);
      putc(bmp[i+2], file);
      putc(bmp[i+3], file);
      putc(bmp[i+4], file);
      putc(bmp[i+5], file);
      putc(bmp[i+6], file);
      putc(bmp[i+7], file);
    }
  fclose(file);
}

void BMPmake(byte_t* bitmap)
{
  // bitmap signature
  bitmap[0] = 'B';
  bitmap[1] = 'M';

  // file size
  bitmap[2] = (HEADER_SIZE + LENGTH) & 0xFF; // 40 + 14 + 12
  bitmap[3] = ((HEADER_SIZE + LENGTH) >> 8) & 0xFF;
  bitmap[4] = ((HEADER_SIZE + LENGTH) >> 16) & 0xFF;
  bitmap[5] = ((HEADER_SIZE + LENGTH) >> 24) & 0xFF;

  // reserved field (in hex. 00 00 00 00)
  int i;
  for( i = 6; i < 10; i++) bitmap[i] = 0;

  // offset of pixel data inside the image
  bitmap[10]=54;
  for( i = 11; i < 14; i++) bitmap[i] = 0;

  // -- BITMAP HEADER -- //

  // header size
  bitmap[14] = 40;
  for( i = 15; i < 18; i++) bitmap[i] = 0;

  bitmap[18] = N & 0xFF;
  bitmap[19] = (N >> 8) & 0xFF;
  bitmap[20] = (N >> 16) & 0xFF;
  bitmap[21] = (N >> 24) & 0xFF;
  bitmap[22] = N & 0xFF;
  bitmap[23] = (N >> 8) & 0xFF;
  bitmap[24] = (N >> 16) & 0xFF;
  bitmap[25] = (N >> 24) & 0xFF;
  // reserved field
  bitmap[26] = 1;
  bitmap[27] = 0;
  // number of bits per pixel
  bitmap[28] = 24; // 3 byte
  bitmap[29] = 0;
  // compression method (no compression here)
  for( i = 30; i < 34; i++) bitmap[i] = 0;

  // size of pixel data
  bitmap[34] = 255; // 12 bits => 4 pixels
  bitmap[35] = 0;
  bitmap[36] = 0;
  bitmap[37] = 0;

  // horizontal resolution of the image - pixels per meter (2835)
  bitmap[38] = 0;
  bitmap[39] = 0;
  bitmap[40] = 48;
  bitmap[41] = 177;

  // vertical resolution of the image - pixels per meter (2835)
  bitmap[42] = 0;
  bitmap[43] = 0;
  bitmap[44] = 48;
  bitmap[45] = 177;

  // color pallette information
  for(i = 46; i < 50; i++) bitmap[i] = 0;

  // number of important colors
  for( i = 50; i < 54; i++) bitmap[i] = 0;

  memset (bitmap + HEADER_SIZE, LENGTH, 0);
}

// should be consuming:
// - an array of chars, that will be the image
// - a FILE struct
// kernel forces every thread to color one character
// to the FILE object

__global__ void cudaColor (byte_t* bmp)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  bmp [3*(col + row * N)] = ((row & 0x20) ? 192 : 64);
  bmp [3*(col + row * N) + 1] = ((col & 0x20) ? 192 : 64);
  bmp [3*(col + row * N) + 2] = ((row & 0x80) || (col & 0x80) ? 192 : 64);
}

int main()
{
  
  byte_t *bmp, *dev_bmp;
  // mallocing space fo the bmp array, that has 3 N*N dimensions
  bmp = (byte_t*)malloc ((HEADER_SIZE + LENGTH) * sizeof (byte_t));

  BMPmake (bmp);
  cudaError_t err;

  err= cudaMalloc ((void**)&dev_bmp, (HEADER_SIZE + LENGTH) * sizeof (byte_t));
  printf("Cuda malloc bmp:%s \n", cudaGetErrorString(err));
		
  err = cudaMemcpy (dev_bmp, bmp, (HEADER_SIZE + LENGTH) * sizeof (byte_t),
		    cudaMemcpyHostToDevice);
  printf("Cuda memcpy to device bmp:%s \n", cudaGetErrorString(err));

  // setting morphed dimensions
  dim3 dimBlock (32, 32);
  dim3 dimGrid (N / dimBlock.x, N / dimBlock.y);

  struct timeval begin, end;
  gettimeofday (&begin, NULL);
  cudaColor <<< dimGrid, dimBlock >>> (dev_bmp + HEADER_SIZE);
  err = cudaPeekAtLastError();
  printf ("Cuda kernel:%s \n", cudaGetErrorString(err));
  gettimeofday (&end, NULL);
 
  err = cudaMemcpy (bmp, dev_bmp, (HEADER_SIZE + LENGTH) * sizeof (byte_t),
		    cudaMemcpyDeviceToHost);
  
  printf("Cuda memcpy to host bmp:%s \n", cudaGetErrorString(err));
		
  BMPwrite(bmp);
  
  int verify = 0,j;
  for(j = 0; j < LENGTH; j++)
    verify += (bmp [j + HEADER_SIZE] == 235);
  printf ("Verify count: %d\n", verify);
  if (verify == (N * N) / 2){
    printf ("Verified!\n");
  } else {
    printf ("pixels not correct\n");
  }
  fprintf (stdout, "time = %lf\n", (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000);
  
  // copying from the device back to the host, time to read out the results
  printf ("size of the image: %d\n", sizeof(bmp));
  
  cudaFree(dev_bmp);
  free(bmp);
  return 0;
}
