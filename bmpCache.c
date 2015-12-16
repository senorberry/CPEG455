#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

#define N 1024
#define length (54+(3*N*N))

#define screenh N
#define screenw N

typedef struct{
  int r;
  int g;
  int b;
} color;

void BMPmake(unsigned char* bitmap)
{
  // -- FILE HEADER -- //

  // bitmap signature
  bitmap[0] = 'B';
  bitmap[1] = 'M';

  // file size
  bitmap[2] = (length & 0xFF); 
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

  // width of the image
  //bitmap[18] = N;
  //for( i = 19; i < 22; i++) bitmap[i] = 0;

  bitmap[18] = N & 0xFF;
  bitmap[19] = (N >> 8) & 0xFF;
  bitmap[20] = (N >> 16) & 0xFF;
  bitmap[21] = (N >> 24) & 0xFF;

  fprintf(stdout,  "1: %d 2: %d 3: %d 4: %d",
	  bitmap[18],
	  bitmap[19],
	  bitmap[20],
	  bitmap[21]);

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

  // -- PIXEL DATA -- //
  for( i = 54; i < length; i+=8) {
    if(i%13==0){bitmap[i] = 0;}
    else{bitmap[i]=235;}
    
    if((i+1)%13==0){bitmap[i+1] = 0;}
    else{bitmap[i+1]=235;}
    
    if((i+2)%13==0){bitmap[i+2] = 0;}
    else{bitmap[i+2]=235;}
    
    if((i+3)%13==0){bitmap[i+3] = 0;}
    else{bitmap[i+3]=235;}
    
    if((i+4)%13==0){bitmap[i+4] = 0;}
    else{bitmap[i+4]=235;}
    
    if((i+5)%13==0){bitmap[i+5] = 0;}
    else{bitmap[i+5]=235;}
    
    if((i+6)%13==0){bitmap[i+6] = 0;}
    else{bitmap[i+6]=235;}

    if((i+7)%13==0){bitmap[i+7] = 0;}
    else{bitmap[i+7]=235;}
  }
}

void BMPwrite(unsigned char* bmp)
{
  int i;
  FILE *file;
  file = fopen("cache.bmp", "w+");

  // trying to see if they can tell me the actual size of the file
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

int main(){
    
  int i=0, j=54;
  unsigned char* bmp=(unsigned char *) malloc(length*sizeof(unsigned char));
  struct timeval begin, end;

  // testing sequence
  for (i =0; i<10; i++){   
    gettimeofday(&begin, NULL);
    BMPmake(bmp);
    BMPwrite(bmp);
    gettimeofday(&end, NULL);
    fprintf(stdout, "time = %lf\n", (end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)*1.0/1000000);
    int test = length-54;
    int verify = 0;

    // verification of the color matrix
    for(j=54; j<length; j++)
      {
	if(j%13==0) {
	  if(bmp[j] == 0) // verifying the non-colored pixel
	    {
	      verify++;
	    }
	}
	else{
	  if(bmp[j] == 235) // verifying the colored pixel
	    {
	      verify++;
	    }
	}
      }
    if (verify == test){
      printf("Verified!\n");
    } else {
      printf("Constants not correct\n");
    }
  }

  /* free(arr); */
  free(bmp);
  return 0;
}
