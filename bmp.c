#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

char bitmap[1000];

int screenh=16;
int screenw=16;

typedef struct {
  int x;
  int y;
  int z;
} vect;


typedef struct{
  int r;
  int g;
  int b;
} color;


typedef struct{
  vect p;
  color c;        
} light;


typedef struct{
  color c;
  int ref;
} mat;


typedef struct {
  int r;
  vect p;
  mat m;
} sphere;

typedef struct{
  vect dir;
  vect orig;
} ray;

vect vectAdd(vect a, vect b) 
{
  vect c;
  c.x=a.x+b.x;
  c.y=a.y+b.y;
  c.z=a.z+b.z;     
  return c;
}        
        
vect vectSub(vect a, vect b)
{
  vect c;
  c.x=a.x-b.x;
  c.y=a.y-b.y;
  c.z=a.z-b.z;     
  return c;
}  

float vectDotProd(vect a, vect b){
  return a.x*b.x+a.y*b.y+a.z*b.z;      
      
}

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
  for( i = 54; i < 822; i++) {
    if(i%2==0){bitmap[i] = 0;}
    else{bitmap[i]=255;}
  }
}

void BMPwrite()
{
  int i;
  FILE *file;
  file = fopen("bitmap.bmp", "w+");
  for(i = 0; i < 822; i++)
    {
      fputc(bitmap[i], file);
    }
  fclose(file);
}


/* mat intersect(ray rayA, sphere S){ */
/*   vect r;  */
/*   vect pc; */
/*   r = rayA.dir; */
/*   //Vect d; */
/*   r= vectUnit(r); */
/*   vect o; */
/*   o=ray.orig; */
/*   pc=s.p; */
	

/*   vect dp; */

/*   dp=vectSub(pc,o); */
/*   double x=pow(vectDotProd(r,dp),2)-(pow(vectMag(dp),2)-pow(S->radius,2)); */
/*   if(x>0){ */
/*     double tp = vectDotProd(r,dp)+sqrt(x); */
/*     double tn = vectDotProd(r,dp)-sqrt(x); */
		
/*     if(tp>=0 || tn>=0 ){ */
/*       inter = make_intersection(); */
/*       vect I; */
/*       if(tp<tn){ */
/* 	inter->t= tp; */
/* 	vectAddS(tp, ray.dir, ray.orig, inter.p); */
/* 	//VectCopy(inter->P, I); */
/*       }else { */
/* 	vectAddS(tn, ray.dir, ray.orig, inter.p); */
/* 	inter->t= tn; */
/* 	//VectCopy(inter->P, I); */
/*       } */
/*       vect poc; */
/*       inter.mat=S.mat; */
/*       poc =vectSub(S.p,ray_cam->center); */
/*       vect n; */
/*       n=vectSub(inter.p,poc); */
/*       vectAddS(1,n,ray.orig,n); */
/*       vectUnit(n); */
/*       //inter->N = n;//not sure if needed */
/*       return inter; */
/*     } */
/*   } */
/*   else{ */
/*     return NULL; */
/*   }     */
/* } */

int main(){
    
  int i,j,k;
  sphere spheres[10];
  for (i=0; i<10; i++){
    spheres[i].p.x = i+2;
    spheres[i].p.y = 14-i;
    spheres[i].p.z = i+2;
    spheres[i].r=4;
    spheres[i].m.c.r=i*255/10;
    spheres[i].m.c.r=(1-i/10)*255;
  }
    
	
  /* for(k=0;k<screenw;k++){ */
  /*   for(j=0;j<screenh;j++){ */
  /*     for (i=0; i<10; i++){ */

  /*     } */
  /*   } */
  /* } */
  struct timeval begin, end;
     
  gettimeofday(&begin, NULL);
  BMPmake();
  BMPwrite();
  gettimeofday(&end, NULL);

  fprintf(stdout, "time = %lf\n", (end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)*1.0/1000000);
    
    
    
  return 0;
}
    
