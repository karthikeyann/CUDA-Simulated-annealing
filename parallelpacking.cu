#define N 10
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<math.h>
#include"functionpacking.cu"
#define ERROR(x) {printf(x); exit(0);}
int main(int argc, char* argv[])
{
	//Read blocks file
	char filename[30];
	sprintf(filename,"SOFT/n%d.cblks",N);
	FILE *f;
	int blkcount=0,hb,sb;
	f=fopen(filename,"r");
	if(f==NULL) ERROR("Can't open Block file\n");
	fscanf(f,"%d %d\n",&sb,&hb);
	printf("%d,%d\n",sb,hb);
	if(filename[0]=='S')	blkcount=sb;
	else blkcount=hb;
	float *w,*h,*area;
	float totalarea=0,aspectratio1,aspectratio2,aspectratio;
	w=(float*)malloc(sizeof(float)*blkcount);
	h=(float*)malloc(sizeof(float)*blkcount);
	area=(float*)malloc(sizeof(float)*blkcount);
	if(filename[0]=='S')
	{
		for(int i=0;i<blkcount;i++)
			{
			fscanf(f,"%f %f %f\n",&area[i],&aspectratio1,&aspectratio2);
			//if(rand()<RAND_MAX/2)
			// aspectratio = (aspectratio1+aspectratio2)/2.0;	//!Average
			//else
			aspectratio = aspectratio1;	//!Average
			w[i]=sqrt(area[i]*aspectratio);
			h[i]=sqrt(area[i]/aspectratio);
			totalarea=totalarea+area[i];
			}
	}
	else
	{
		for(int i=0;i<blkcount;i++)
			{
			fscanf(f,"%f %f %f\n",&area[i],&w[i],&h[i]);
			totalarea=totalarea+area[i];
			}
	}
	fclose(f);
	//Read Permutations
	int n=blkcount,s=blkcount;
	int *permutations;//[s][n];
	permutations=(int*)malloc(sizeof(int)*n*s);
	sprintf(filename,"n%ds%d.perms",N,N);
	f=fopen(filename,"r");
	if(f==NULL) ERROR("Can't open Perm file\n");
	for(int i=0;i<s;i++)
		for(int j=0;j<n;j++)
			fscanf(f,"%d",&permutations[i*n+j]);
	fclose(f);
	for(int i=0;i<s;i++){
		for(int j=0;j<n;j++)
			printf("%d ",permutations[i*n+j]);
			printf("\n");
	}
	//*/HOST Lencalc test
	float xpos[N];
	printf("Host Wid=%f\n",lencalc(permutations, permutations, w, xpos, 1));
	//*/
	//*/
	float *returnArea;
	float *d_w, *d_h, *d_returnArea;
	int *d_perms, *h_perms;
	long *seed;
	returnArea=(float*)malloc(sizeof(float)*N*N);
	h_perms=(int*)malloc(sizeof(int)*n*s);
	/*/DATA to COPY to GPU
	w n*float
	h n*float
	permutations n*s*int
	totalarea
	//*/
	cudaMalloc((void**)&d_w,sizeof(float)*N);
	cudaMalloc((void**)&d_h,sizeof(float)*N);
	cudaMalloc((void**)&d_returnArea,sizeof(float)*N*N);
	cudaMalloc((void**)&d_perms, sizeof(int)*n*s);
	cudaMalloc((void**)&seed,sizeof(long)*N*N);
	
	cudaMemset(seed,rand(),sizeof(long)*N*N);
	cudaMemcpy(d_w,w,sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_h,h,sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_perms,permutations,sizeof(int)*n*s, cudaMemcpyHostToDevice);
	annealing<<<N,N>>>(d_w, d_h, d_perms , seed, totalarea, d_returnArea);
	cudaMemcpy(returnArea,d_returnArea,sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	printf("\n Area GPU calc\n");
	float mini=99999999; 	for(int i=0;i<N*N;i++) mini=mini<returnArea[i]?mini:returnArea[i];
	for(int i=0;i<N*N;i++)
		printf("%f, ",returnArea[i]);
	printf("\n");
	printf("Minim=%f\n",mini);
	printf("TotlA=%f\n",totalarea);
	printf("Deads=%f\n",(mini-totalarea)*100.0/mini); //Deads=8.268910 for n10s10, Deads=8.085245, time=2.440s
	cudaMemcpy(h_perms,d_perms,sizeof(int)*n*s, cudaMemcpyDeviceToHost);
		printf("\n Test\n");
	for(int i=0;i<s;i++){
		for(int j=0;j<n;j++)
			printf("%d ",h_perms[i*n+j]);
		printf("\n");
	}
	cudaFree(d_w);
	cudaFree(d_h);
	cudaFree(d_returnArea);
	cudaFree(d_perms);
	cudaFree(seed);
	free(w); free(h); free(area); free(permutations); free(returnArea); free(h_perms);
	printf("%d %d %d\n\n\t",blkcount,n,s);
	return 0;
}
