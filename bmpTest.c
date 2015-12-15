#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>


void BMPread()
{
  int i;
  FILE *file1;
  file1 = fopen("cache.bmp", "r");
  FILE *file2;
  file2 = fopen("cuda.bmp", "r");
 char c; 
 while(c=fgetc(file1)!= EOF)
    {
	assert(c==fgetc(file2));      
    }
   


  fclose(file1);

  fclose(file2);
}



int main(){
    
  
	 BMPread();


  return 0;
}
