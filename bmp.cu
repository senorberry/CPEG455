#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

#define N 1024
#define length 54+(3*N*N)

#define screenh N
#define screenw N

void BMPwrite(unsigned char* bmp)
{
  int i;
  FILE *file;
  file = fopen("cuda.bmp", "w+");
  for(i = 0; i < length; i+=8)
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

void BMPmake(unsigned char* bitmap)
{
  // bitmap signature
  bitmap[0] = 'B';
  bitmap[1] = 'M';

  // file size
  bitmap[2] = length & 0xFF; // 40 + 14 + 12
  bitmap[3] = (length >> 8) & 0xFF;
  bitmap[4] = (length >> 16) & 0xFF;
  bitmap[5] = (length >> 24) & 0xFF;

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

  // THIS IS BEING OFFLOADED TO THE GPU
  // -- PIXEL DATA -- 
  for( i = 54; i < length; i++) {
    bitmap[i] = 0;
  }
}

// should be consuming:
// - an array of chars, that will be the image
// - a FILE struct
// kernel forces every thread to color one character
// to the FILE object

__global__ void cudaColor(unsigned char *bmp)
{ // only one block, whose dimension is half the length
  int col = threadIdx.x;
  int row = threadIdx.y;
  // the 54 is necessary because of the image offset that is being
  // applied due to the format of the bitmap
  if ((row*(length/2)+col)<54)
    return;
  bmp[(row*(length/2))+col] = 1;
  
  // initializes the value at the position
  if(((row*(length/2))+col)%2==0)
    {
      bmp[(row*(length/2))+col] = 1;
    }
  else
    {
      bmp[(row*(length/2))+col] = 235;
    }
}

int main()
{
  unsigned char *bmp, *dev_bmp; 
  // mallocing space fo the bmp array, that has 3 N*N dimensions
  bmp = (unsigned char *) malloc(length*sizeof(unsigned char));

  // cudaMalloc on the device
  cudaError_t
    err = cudaMalloc((void**)&dev_bmp, (length)*sizeof(unsigned char));
  printf("Cuda malloc bmp:%s \n", cudaGetErrorString(err));
		
  // inits the memory blocks
  BMPmake(bmp);

  err = cudaMemcpy(dev_bmp, bmp,
		   (length)*sizeof(unsigned char), cudaMemcpyHostToDevice);
  printf("Cuda memcpy to device bmp:%s \n", cudaGetErrorString(err));

  // setting morphed dimensions
  dim3 dimBlock((length-54)/2, (length-54)/2);
  dim3 dimGrid(1, 1);
  // timing the operation

  struct timeval begin, end;
  gettimeofday(&begin, NULL);
  // copy the information to the device from the host
  cudaColor <<< dimGrid, dimBlock >>>(dev_bmp);
  gettimeofday(&end, NULL);
 
  // copy data back
  err = cudaMemcpy(bmp, dev_bmp,
		   (length)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  
  printf("Cuda memcpy to host bmp:%s \n", cudaGetErrorString(err));
		
  BMPwrite(bmp);
  int verify = 0, test = length-54,j;
  for(j=54; j<length; j++)
    {
      if(j%13==0) {
	printf("value: %d\n",bmp[j]);
	if(bmp[j] == 1) // verifying the non-colored pixel
	  {
	    printf("value: %d\n",bmp[j]);
	    verify++;
	  }
      }else{
	if(bmp[j] == 235) // verifying the colored pixel
	  {
	    verify++;
	  }
      }
    }
  if (verify == test){
    printf("Verified!\n");
  } else {
    printf("pixels not correct\n");
  }
  fprintf(stdout, "time = %lf\n", (end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)*1.0/1000000);
		
  // copying from the device back to the host, time to read out the results
  printf("size of the image: %d\n", sizeof(bmp));
  cudaFree(dev_bmp);
  free(bmp);
  return 0;
}
