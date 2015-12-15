#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>


char bitmap[1000];

int screenh=16;
int screenw=16;

void BMPmake()
{
  // -- FILE HEADER -- //

  // bitmap signature
  bitmap[0] = 'B';
  bitmap[1] = 'M';

  // file size
  bitmap[2] = 255; // 40 + 14 + 12
  bitmap[3] = 3;
  bitmap[4] = 0;
  bitmap[5] = 0;

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
  bitmap[18] = 16;
  for( i = 19; i < 22; i++) bitmap[i] = 0;

  // height of the image
  bitmap[22] = 16;
  for( i = 23; i < 26; i++) bitmap[i] = 0;

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
  // where the coloration takes place
  for( i = 54; i < 822; i++) {
    if(i%2==0){bitmap[i] = 0;}
    else{bitmap[i]=174;}
  }
}

void BMPwrite()
{
  int i;
  FILE *file;
  file = fopen("baseline.bmp", "w+");
  for(i = 0; i < 822; i++)
    {
      fputc(bitmap[i], file);
    }
  fclose(file);
}

int main(){
    
  struct timeval begin, end;
     
  gettimeofday(&begin, NULL);
  BMPmake();
  BMPwrite();
  gettimeofday(&end, NULL);

  fprintf(stdout, "time = %lf\n", (end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)*1.0/1000000);
    
  return 0;
}
    
