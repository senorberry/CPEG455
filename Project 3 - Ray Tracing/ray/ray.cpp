#include <stdio.h>
#include <stdlib.h>

#include "udray.h"
#include "glm.h"

bool gray=false;
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

extern Camera *ray_cam;       // camera info
extern int image_i, image_j;  // current pixel being shaded
extern bool wrote_image;      // has the last pixel been shaded?

// reflection/refraction recursion control

extern int maxlevel;          // maximum depth of ray recursion 
extern double minweight;      // minimum fractional contribution to color

// these describe the scene

extern vector < GLMmodel * > model_list;
extern vector < Sphere * > sphere_list;
extern vector < Light * > light_list;

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//grayscale shade

void shade_ray_gray(Ray *ray, Intersection *inter, Vect color)
{
	Vect L;
	double diff_factor;

	// iterate over lights
	double maxf=10;
	for (int i = 0; i < light_list.size(); i++) {

		// AMBIENT

		
		double f =(inter->t/maxf);
		if(f>maxf){maxf=f;}
		//f= maxf;
		color[R] += f; //* light_list[i]->amb[R];
		color[G] += f; //* light_list[i]->amb[G];
		color[B] += f; //* light_list[i]->amb[B];
	}
	VectClamp(color, 0, 1);
}



// intersect a ray with the entire scene (.obj models + spheres)

// x, y are in pixel coordinates with (0, 0) the upper-left hand corner of the image.
// color variable is result of this function--it carries back info on how to draw the pixel

void trace_ray(int level, double weight, Ray *ray, Vect color)
{
	Intersection *nearest_inter = NULL;
	Intersection *inter = NULL;
	int i;

	// test for intersection with all .obj models

	for (i = 0; i < model_list.size(); i++) {
		inter = intersect_ray_glm_object(ray, model_list[i]);
		if(inter!=NULL){
			double y =0;
		}
		update_nearest_intersection(&inter, &nearest_inter);
	}

	// test for intersection with all spheres

	for (i = 0; i < sphere_list.size(); i++) {
		inter = intersect_ray_sphere(ray, sphere_list[i]);
		update_nearest_intersection(&inter, &nearest_inter);
	}

	// "color" the ray according to intersecting surface properties

	// choose one of the simpler options below to debug or preview your scene more quickly.
	// another way to render faster is to decrease the image size.

	if (nearest_inter) {
		//shade_ray_false_color_normal(nearest_inter, color);
		//    shade_ray_intersection_mask(color);  
		//shade_ray_diffuse(ray, nearest_inter, color);
		//shade_ray0_local(ray, nearest_inter, color);
		 //  shade_ray_recursive(0/*level*/, weight, ray, nearest_inter, color);
		if(gray){shade_ray_gray(ray, nearest_inter, color);}
		else{ shade_ray_recursive(2/*level*/, weight, ray, nearest_inter, color);}
	}


	// color the ray using a default

	else
		shade_ray_background(ray, color); 
}

//----------------------------------------------------------------------------

// test for ray-sphere intersection; return details of intersection if true

Intersection *intersect_ray_sphere(Ray *ray, Sphere *S)
{
	// FILL IN CODE (line below says "no" for all spheres, so replace it)
	Intersection *inter;

	Vect r; 
	Vect pc;
	VectCopy(r, ray->dir);
	//Vect d;
	VectUnit(r);
	Vect o;
	VectCopy(o,ray->orig);
	VectCopy( pc,S->P);
	
	//VectSub(pc, o, pc);
	Vect dp;
	//Vect no;
	//VectNegate(o,no);
	VectSub(pc,o,dp);
	double x=pow(VectDotProd(r,dp),2)-(pow(VectMag(dp),2)-pow(S->radius,2));
	if(x>0){
		double tp = VectDotProd(r,dp)+sqrt(x);
		double tn = VectDotProd(r,dp)-sqrt(x);
		
		if(tp>=0 || tn>=0 ){
			inter = make_intersection();
			Vect I;
			if(tp<tn){
				inter->t= tp;
				VectAddS(tp, ray->dir, ray->orig, inter->P);
				//VectCopy(inter->P, I);
			}else {
				VectAddS(tn, ray->dir, ray->orig, inter->P);
				inter->t= tn;
				//VectCopy(inter->P, I);
			}
			Vect poc;
			inter->surf=S->surf;
			VectSub(S->P,ray_cam->center, poc);
			Vect n;
			VectSub(inter->P,poc,n);
			VectAddS(1,n,ray->orig,n);
			VectUnit(n);
			VectCopy(inter->N , n);
			//inter->N[Y] = 1;
			//inter->N[Z] = 1;
			return inter;
		}
	
	}
	else{
		return NULL;
	}

	//inter = make_intersection();

	//Vect p = S->P;
	/*
	Vect p;
	VectCopy(p,S->P);
	Vect r; 
	VectCopy(r,ray->dir);
	double theta = acos(VectDotProd(p,r)/(VectMag(p)*VectMag(r))); //is acos necessary?
	*/
	//             A dot B 
	// arccos (  ----------- )  = theta
	//           ||A||*||B||
	// p-camerapos=vector camera to sphere, pc
	// pc  + radius_|_ = Raduis max
	// 
	/*Vect pc;
	VectSub(p,ray->orig,pc);
	double thetamax = atan( S->radius/VectMag(pc));
	
	//if correct
	if(theta>thetamax){
		inter = make_intersection();
		
	
	}*/

	//
	//	acos(VectDotProd(pc,rm)/(VectMag(pc)*VectMag(rm)))



	return NULL;

}

//----------------------------------------------------------------------------

// only local, ambient + diffuse lighting (no specular, shadows, reflections, or refractions)

void shade_ray_diffuse(Ray *ray, Intersection *inter, Vect color)
{
	Vect L;
	double diff_factor;

	// iterate over lights
	double maxf=0;
	for (int i = 0; i < light_list.size(); i++) {

		// AMBIENT

		//printf("1:%d \n", color[R]);

		color[R] += inter->surf->amb[R] * light_list[i]->amb[R];
		color[G] += inter->surf->amb[G] * light_list[i]->amb[G];
		color[B] += inter->surf->amb[B] * light_list[i]->amb[B];

			VectClamp(color, 0, 1);
		//printf("2:%d \n", color[R]);
		/*double f =(pow(inter->t/2-1,2));
		if(f>maxf){maxf=f;}
		color[R] += f; //* light_list[i]->amb[R];
		color[G] += f; //* light_list[i]->amb[G];
		color[B] += f; //* light_list[i]->amb[B];
		//printf("3:%e \n", color[R]);
		*/
		// DIFFUSE 

		// FILL IN CODE
		Vect l;
		VectSub(inter->P,light_list[i]->P,l);
		VectUnit(l);
		VectNegate(l, l);
		double dif = VectDotProd(inter->N,l);
		
		if(dif>0 ){

			color[R] += inter->surf->diff[R] * light_list[i]->diff[R]*dif;
			color[G] += inter->surf->diff[G] * light_list[i]->diff[G]*dif;
			color[B] += inter->surf->diff[B] * light_list[i]->diff[B]*dif;
		}
	}

	// clamp color to [0, 1]
	//if(maxf!=0){
		//printf("Max: %e", maxf);}
	VectClamp(color, 0, 1);
}


//----------------------------------------------------------------------------

// same as shade_ray_diffuse(), but add specular lighting + shadow rays (i.e., full Phong illumination model)

Intersection *shadow_sphere_intersect(Ray * ray, Sphere * S){

	Intersection *inter;

	
	Vect r; 
	Vect pc;
	VectCopy(r, ray->dir);
	//Vect d;
	//VectUnit(r);
	Vect o;
	VectCopy(o,ray->orig);
	VectCopy( pc,S->P);
	
	VectSub(pc, o, pc);
	Vect dp;
	//Vect no;
	//VectNegate(o,no);
	VectSub(pc,o,dp);
	double x=pow(VectDotProd(r,dp),2)-(pow(VectMag(dp),2)-pow(S->radius,2));
	if(x>0){
		double tp = VectDotProd(r,dp)+sqrt(x);
		double tn = VectDotProd(r,dp)-sqrt(x);
		
		if(tp>=0 || tn>=0 ){
			inter = make_intersection();
			Vect I;
			if(tp<tn){
				inter->t= tp;
				VectAddS(tp, ray->dir, ray->orig, I);
				VectCopy(inter->P, I);
			}else {
				VectAddS(tn, ray->dir, ray->orig, I);
				inter->t= tn;
				VectCopy(inter->P, I);
			}
			Vect poc;
			inter->surf=S->surf;
			VectSub(S->P,ray->orig/*ray_cam->center*/, poc);
			Vect n;
			VectSub(inter->P, poc,n);
			VectAddS(1,n,ray->orig,n);
			VectUnit(n);
			VectCopy(inter->N , n);
			//inter->N[Y] = 1;
			//inter->N[Z] = 1;
			return inter;
		}
	
	}
	else{
		return NULL;
	}

	


	return NULL;
}


void shade_ray_local(Ray *ray, Intersection *inter, Vect color)
{

	// FILL IN CODE 


	shade_ray_diffuse(ray, inter, color);
	

	
	for (int i = 0; i < light_list.size(); i++) {
	

		
		Vect view,ref,dir,n;
		VectCopy(n,inter->N);
		//VectUnit(n);
		//n[3]=-n[3];
		//VectNegate(n, n);
		VectCopy(view, ray_cam->eye);
		VectSub(view, inter->P, view);
		VectUnit(view);
		//VectNegate(view, view);

		VectCopy(dir,light_list[i]->P);
		//VectNegate(dir,dir);
		VectSub(dir,inter->P,ref);
		//VectNegate(dir,dir);
		//VectSub(inter->P, dir , ref);
		if(ref[0]==0 && ref[1]==0 && ref[2]==0){
		
		}
		else{VectUnit(ref);
		}
		double d = -(2*VectDotProd(ref,n));
		//if(d>0){
		VectAddS(d,n,ref,ref);
		
		//VectAddS(2,n,ref,ref);
		if(ref[0]==0 && ref[1]==0 && ref[2]==0){

		}
		else{
			//VectUnit(ref);
		}
		double s;
		double g = VectDotProd(ref,view);
		if(g<0){
			g = 0 - g;
			s = pow(g,inter->surf->spec_exp);}
		else{s=0;}
		/*double Red = inter->surf->spec[R] * light_list[i]->spec[R] *s ;
		if(Red != 0){
			printf("not");
		}*/
		//double r = inter->surf->spec[R] * light_list[i]->spec[R] *s;
		color[R] += inter->surf->spec[R] * light_list[i]->spec[R] *s;
		color[G] += inter->surf->spec[G] * light_list[i]->spec[G] *s;
		color[B] += inter->surf->spec[B] * light_list[i]->spec[B] *s;
		
	}

	Intersection *shadow;	
	//shadows
	shadow = NULL;
	for (int i = 0; i < light_list.size(); i++){
	
		Ray *shadeRay;
		shadeRay = make_ray();
		VectCopy(shadeRay->orig,inter->P);
		Vect shade;
		VectSub(light_list[i]->P, inter->P,shade);
		VectUnit(shade);
		VectCopy(shadeRay->dir,shade);
		VectAddS(0.001,inter->N,shadeRay->orig,shadeRay->orig);

	for (int j = 0; j < model_list.size(); j++) {
		//shadow = intersect_ray_glm_object(shadeRay, model_list[j]);
		//update_nearest_intersection(&inter, &nearest_inter);
		if(shadow !=NULL){
		break;}
	}

	// test for intersection with all spheres
	if(shadow == NULL){
		for (int j = 0; j < sphere_list.size(); j++) {
		
			//shadow = shadow_sphere_intersect(shadeRay,sphere_list[j]);
			//shadow = intersect_ray_sphere(shadeRay, sphere_list[j]);
			if(shadow !=NULL){
				break;}

	



	
		}
	}
		if(shadow != NULL){
			//printf("X: %e Y: %e Z: %e",shadow->P[0], shadow->P[1], shadow->P[2]);
			color[R] = 0; //inter->surf->spec[R] * light_list[i]->spec[R] *0.5;
			color[G] = 0; //inter->surf->spec[G] * light_list[i]->spec[G] *0.5;
			color[B] = 0; //inter->surf->spec[B] * light_list[i]->spec[B] *0.5;	
		}
	}
	

		
	
	

	VectClamp(color, 0, 1);

}

//----------------------------------------------------------------------------

// full shading model: ambient/diffuse/specular lighting, shadow rays, recursion for reflection, refraction

// level = recursion level (only used for reflection/refraction)

void shade_ray_recursive(int level, double weight, Ray *ray, Intersection *inter, Vect color)
{
	Surface *surf;
	int i;

	// initialize color to Phong reflectance model
		Ray *refRay= make_ray();;
		VectCopy(refRay->orig, inter->P);

		Vect v;
		//VectSub(inter->P,ray_cam->eye, v);
		VectCopy(v,ray->dir);
		VectUnit(v);

		double nv = VectDotProd(inter->N,v);
		VectAddS(-2*nv,inter->N,v,refRay->dir);
		VectAddS(0.1, inter->N,refRay->orig, refRay->orig);

				Intersection *newInter = NULL;
				Intersection *near = NULL;

				for (i = 0; i < model_list.size(); i++) {
					newInter = intersect_ray_glm_object(refRay, model_list[i]);
					update_nearest_intersection(&newInter, &near);
				}

	// test for intersection with all spheres

				for (i = 0; i < sphere_list.size(); i++) {
					newInter = intersect_ray_sphere(refRay, sphere_list[i]);
					update_nearest_intersection(&newInter, &near);
				}

			if(true){		
				shade_ray_local(ray, inter, color);
			}

	// if not too deep, recurse
		if(near){
				Surface t = *inter->surf;
				surf=&t;
				
	if (level + 1 < maxlevel) {

		// add reflection component to color

		if (surf->reflectivity * weight > minweight) {

			// FILL IN CODE



			
			
			
			
			
				shade_ray_recursive(level+1, weight, refRay, near,  color);
			
			//shade_ray_local(refRay,inter,color);


		}

		// add refraction component to color

		if (surf->transparency * weight > minweight) {

			// GRAD STUDENTS -- FILL IN CODE

		}
		}
	}
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// ray trace another pixel if the image isn't finished yet

void idle()
{
	if (image_j < ray_cam->im->h) {

		raytrace_one_pixel(image_i, image_j);

		image_i++;

		if (image_i == ray_cam->im->w) {
			image_i = 0;
			image_j++;
		}    
	}

	// write rendered image to file when done

	else if (!wrote_image) {

		write_PPM("output.ppm", ray_cam->im);

		wrote_image = true;
	}

	glutPostRedisplay();
}

//----------------------------------------------------------------------------

// show the image so far

void display(void)
{
	// draw it!

	glPixelZoom(1, -1);
	glRasterPos2i(0, ray_cam->im->h);

	glDrawPixels(ray_cam->im->w, ray_cam->im->h, GL_RGBA, GL_FLOAT, ray_cam->im->data);

	glFlush ();
}

//----------------------------------------------------------------------------

void init()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, ray_cam->im->w, 0.0, ray_cam->im->h);
}

//----------------------------------------------------------------------------

int main(int argc, char** argv)
{
	glutInit(&argc, argv);

	// initialize scene (must be done before scene file is parsed)

	init_raytracing();

	if (argc == 2)
		parse_scene_file(argv[1], ray_cam);
	else {
		printf("missing .scene file\n");
		exit(1);
	}

	// opengl business

	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(ray_cam->im->w, ray_cam->im->h);
	glutInitWindowPosition(500, 300);
	glutCreateWindow("hw3");
	init();

	glutDisplayFunc(display); 
	glutIdleFunc(idle);

	glutMainLoop();

	return 0; 
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
