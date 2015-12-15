#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>



int N=300;
int length=54+3*N*N;
char *bmp; 

int screenh=N;
int screenw=N;



typedef struct{
  int r;
  int g;
  int b;
} color;



void BMPmake(char* bitmap)
{
  // -- FILE HEADER -- //

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

  // width of the image
  //bitmap[18] = N;
  //for( i = 19; i < 22; i++) bitmap[i] = 0;

  bitmap[18] = N & 0xFF;
  bitmap[19] = (N >> 8) & 0xFF;
  bitmap[20] = (N >> 16) & 0xFF;
  bitmap[21] = (N >> 24) & 0xFF;



  // height of the image
  //bitmap[22] = N;
  //for( i = 23; i < 26; i++) bitmap[i] = 0;

  bitmap[22] = N & 0xFF;
  bitmap[23] = (N >> 8) & 0xFF;
  bitmap[24] = (N >> 16) & 0xFF;
  bitmap[35] = (N >> 24) & 0xFF;



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
  for( i = 54; i < length; i++) {
    
	//pixels are written here
    if(i%2==0){bitmap[i] = 0;}
    else{bitmap[i]=255;}
  }
}

void BMPwrite()
{
  int i;
  FILE *file;
  file = fopen("bitmap.bmp", "w+");
  for(i = 0; i < length; i++)
    {
      putc(bmp[i], file);
    }
  fclose(file);
}


__global__ void bmpCUDA(int *ary, int N)
{
//	int i;
	ary[threadIdx.x]*=2;





}


int main(){
    
  int i;
  bmp=(char *) malloc(length*sizeof(char));
   
	
  /* for(k=0;k<screenw;k++){ */
  /*   for(j=0;j<screenh;j++){ */
  /*     for (i=0; i<10; i++){ */

  /*     } */
  /*   } */
  /* } */
  struct timeval begin, end;

int *arr;
//int N =1000;
arr = (int *) malloc( length*sizeof(int));


  for (i=0;i<10;i++){
     
  gettimeofday(&begin, NULL);

  bmpCUDA<<<1, length>>>(arr, length);
 // BMPwrite();
  gettimeofday(&end, NULL);

  fprintf(stdout, "time = %lf\n", (end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)*1.0/1000000);
 }

for (i =0; i<10; i++){   
    
 	 gettimeofday(&begin, NULL);
	 BMPmake(bmp);
	//BMPwrite();
 	 gettimeofday(&end, NULL);

	  fprintf(stdout, "time = %lf\n", (end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)*1.0/1000000);
     }

	free(arr);
	free(bmp);
  return 0;
}
