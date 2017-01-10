#define FALSE 0
#define TRUE  1
__global__ void rand_init(long *a,long seed)
{
int tid=threadIdx.x+blockDim.x*blockIdx.x;
//long a = 100001;
a[tid] = seed + tid;
}
// returns random integer from 1 to lim
__device__ int rand1(long *a, int lim)
{
int tid=threadIdx.x+blockDim.x*blockIdx.x;
a[tid] = (a[tid] * 125) % 2796203;
return ((a[tid] % lim) + 1);
}
// returns random integer from 1 to lim (Gerhard's generator)
__device__ int rand2(long *a, int lim)
{
int tid=threadIdx.x+blockDim.x*blockIdx.x;
a[tid] = (a[tid] * 32719 + 3) % 32749;
return ((a[tid] % lim) + 1);
}
// returns random integer from 1 to lim (Bill's generator)
__device__ int rand3(long *a, int lim)
{
int tid=threadIdx.x+blockDim.x*blockIdx.x;
a[tid] = (((a[tid] * 214013L + 2531011L) >> 16) & 32767);
return ((a[tid] % lim) + 1);
}

__host__ __device__ float lencalc(int *sx, int *sy, float *w, float *xpos, bool axis)
{
	//int cost=0;
	auto int matchy[N];//,matchx[N];
	auto float L[N];
	for(int i=0;i<N;i++) {
	 //matchx[sx[i]-1]=i;
	 matchy[sy[i]-1]=i;
	 L[i]=0; xpos[i]=0;
	  }
	int b,p;
	float t;
	for(int i=0;i<N;i++){
		if(axis) b=sx[i]-1;
		else b=sx[N-1-i]-1;
		p=matchy[b];
		xpos[b]=L[p];
		t=xpos[b]+w[b];
		for(int j=p; j<N && t>L[j] ;j++) L[j]=t;
	}
	return L[N-1];
}
__device__ void neighbourhood(int *sx,int *sy, int *sxnew, int *synew, long *seed){
	int randxi=rand1(seed,N-1),randxj=rand1(seed,N-1); //In 1D gridding, use N-1,N. for 2D gridding use N-1,N-1	
	int randyi=rand1(seed,N  ),randyj=rand1(seed,N  ); //for 1D N, 2D N-1
	//swap
	for(int i=0;i<N;i++){
		sxnew[i]=sx[i]; synew[i]=sy[i];
	}
	sxnew[randxi]=sx[randxj];
	sxnew[randxj]=sx[randxi];
	synew[randxi]=sy[randxj];
	synew[randxj]=sy[randxi];
}
__device__ void newneighbourhood(int *sx,int *sy, int *sxnew, int *synew, long *seed){
	int randxi=rand1(seed,N-1),randxj=rand1(seed,N-1); //In 1D gridding, use N-1,N. for 2D gridding use N-1,N-1	
	int randyi=rand1(seed,N  ),randyj=rand1(seed,N  ); //for 1D N, 2D N-1
	int choice=rand1(seed,3  );
	for(int i=0;i<N;i++){
		sxnew[i]=sx[i]; synew[i]=sy[i];
	}
	if(choice==1){
		//swap sx
		sxnew[randxi]=sx[randxj];
		sxnew[randxj]=sx[randxi];		
	}
	else if(choice==2){
		synew[randxi]=sy[randxj];
		synew[randxj]=sy[randxi];
	}
	else{
		sxnew[randxi]=sx[randxj];
		sxnew[randxj]=sx[randxi];
		synew[randxi]=sy[randxj];
		synew[randxj]=sy[randxi];
	}
}

__global__ void annealing(float *w, float *h, int *perms , long *seed, int totalarea, float *returnArea){
	int tid=threadIdx.x, uid=blockIdx.x;
	int index1d=threadIdx.x+blockDim.x*blockIdx.x;
	__shared__ float minarea[N];
	__shared__ int minid[N];
	auto int sxA[N], syA[N], sxnewA[N], synewA[N]; //Arrays
	auto int *sx=sxA, *sy=syA, *sxnew=sxnewA, *synew=synewA; //pointer seperately declared to tackle lvalue error
	for(int i=0;i<N;i++){
		sx[i]=perms[uid*N+i];
		sy[i]=perms[tid*N+i]; //use uid, if 1D grid, else 2D grid
		//perms[uid*N+i]=-sy[i];
	}
	float t0=0.00000001, ts=0.01,T,delta; //dead space %4.868905504 in python, %8.26 in CUDA.
	int accepted=0;
	float Wid=0,Len=0,Area=0;
	auto float xpos[N],ypos[N];
	float sdWid=0,sdLen=0,sdArea=0;
	float minsofar=0;
	Wid=lencalc(sx,sy,w,xpos,FALSE); //X-axis
	Len=lencalc(sx,sy,h,ypos,TRUE ); //Y-axis
	Area=Wid*Len;
	returnArea[index1d]=Area;
	for(int ancnt=0;ancnt<5000;ancnt++) //20000
	{
	newneighbourhood(sx,sy,sxnew,synew,seed);
	sdWid=lencalc(sxnew,synew,w,xpos,FALSE); //X-axis
	sdLen=lencalc(sxnew,synew,h,ypos,TRUE ); //Y-axis
	sdArea=sdWid*sdLen;

	auto int *temp;
	if(sdArea<Area){
		temp=sx;
		sx=sxnew;
		sxnew=temp;
		temp=sy;
		sy=synew;
		synew=temp;
		Area=sdArea;
		accepted=accepted+1;
		minsofar=Area;
	}
	else{ //take risk (uses random fn)
		float p= float(rand1(seed,N))/float(N);
		T = 1.0/(t0+ts*accepted);
		delta=sdArea-Area;
		if(p<exp(-delta/T)){
			temp=sx;
			sx=sxnew;
			sxnew=temp;
			temp=sy;
			sy=synew;
			synew=temp;
			Area=sdArea;
			accepted=accepted+1;
		}
	}
	}
	minarea[threadIdx.x]=minsofar;
	minid[threadIdx.x]=threadIdx.x;
	for(int ruled=N/2;ruled>0;ruled/=2)
		if(threadIdx.x<ruled)
			if(minarea[threadIdx.x]>minarea[threadIdx.x+ruled]){
				minarea[threadIdx.x]=minarea[threadIdx.x+ruled];
				minid[threadIdx.x]=minid[threadIdx.x+ruled];
				}
	if(threadIdx.x==minid[0])
	for(int i=0;i<N;i++){
		perms[uid*N+i]=sx[i];
		//perms[uid*N+i]=sy[i]; //use uid, if 1D grid, else tid for 2D grid 
	}
	returnArea[index1d]=minsofar;
}
