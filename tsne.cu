/* William Dreese + Steven Gschwind
 * t-SNE C baseline for mini project
 * and CUDA portion
 *
 * still need to debug and unit test
 * start cuda implementation
 * - look into memory issues
 *
 * goose is doing input (start -> data)
 * have goose do output (sols -> 2D pic)
 */


#include <stdio.h>
#include <math.h>
#include <cuda.h>

int dD, dP;
float perp, l_r, momemtum;

// 2D coord of a data point
struct sol {
	float x,y;
};

//shell function that uses floats with math.h's exp function
float expo(float val){
	return (float)exp((double)val);
}

//find euclidean distance for two data points with arbitrary dimensions
float euclidean_dist(float *xi, float *xj){

	int i;
	float total = 0;

	for (i = 0; i < dP; i++) total += (xi[i]-xj[i])*(xi[i]-xj[i]);
	
	return sqrt(total);

}

//compute true similarity scores between data points
void compute_pij(float *data, float *pij_grid){

	int i, j;
	float total_prob, val;

	for (i = 0; i < dD; i++){
		total_prob = 0;
		for (j = 0; j < dD; j++){
			if (i == j) pij_grid[i*dD+j] = 0;
			else {
				val = expo((0-euclidean_dist(&data[i*dP], &data[j*dP], dP))/(2*perp*perp))
				total_prob += val;
				pij_grid[i*dD+j] = val;
			}
		}
		for (j = 0; j < dD; j++) pij_grid[i*dD+j] /= total_prob; 
	}

}

//pre-processing of pij grid, saves time and increases visual accuracy
void symmetric_pij(float *pij_grid){
	
	int i, j;
	float val;

	for (i = 0; i < dD; i++){
		for (j = i+1; j < dD; j++){
			val = (pij_grid[i*dD+j] + pij_grid[j*dD+i]) / (float)(2*dD);
			pij_grid[i*dD+j] = val;
			pij_grid[j*dD+i] = val;	
		}
	}

}

//each potential solution is randomized 0 < x,y < 1
void random_solutions(sol *sols){
	
	int i;
	
	for (i = 0; i < dD; i++){
		sols[i]->x = (float)rand() / (float)RAND_MAX;
		sols[i]->y = (float)rand() / (float)RAND_MAX;
	}

}

//calculate euclidean distance between two 2D points
float sol_ed(sol *i, sol *j){

	return sqrt(((i->x - j->x)*(i->x - j->x))+((i->y - j->y)*(i->y - j->y)));

}

//calculate low-dimensionality similarity grid
void compute_qij(sol *sols, float *qij_grid){

	int i, j;
	float total, val;

	for (i = 0; i < dD; i++){
		total = 0;
		for (j = 0; j < dD; j++){
			if (i == j) qij_grid[i*dD+j] = 0;
			else {
				val = 1.0 / (1.0+sol_ed(sols[i], sols[j]));
				total += val;
				qij_grid[i*dD+j] = val;	
			}
		}
		for (j = 0; j < dD; j++) qij_grid[i*dD+j] /= total;
	}
}

//calculates and applies gradients to each solution
void compute_gradients(sol *sols, float *pij_grid, float *qij_grid){

	int i, j;
	float gradX, gradY, prevX, prevY, pq, ed;

	for (i = 0; i < dD; i++){
		gradX = 0; gradY = 0;
		for (j = 0; j < dD; j++){
			pq = (pij_grid[i*dD+j] - qij_grid[i*dD+j]);
			ed = 1.0 / (1.0 + sol_ed(sols[i],sols[j]));
			gradX += pq * (sol[i]->x - sol[j]->x) * ed;
			gradY += pq * (sol[i]->y - sol[j]->y) * ed;
		}
		gradX *= 4; gradY *= 4;
		prevX = sols[i]->x; prevY = sols[i]->y;
		sols[i]->x += l_r*gradX + momemtum*(sols[i]->x - prev_sols[i]->x);
		sols[i]->y += l_r*gradY + momemtum*(sols[i]->y - prev_sols[i]->y);
		prev_sols[i]->x = prevX; prev_sols[i]->y = prevY;
	}

}

//honestly, the exact same as baseline compute_pij, except each row is sent to it's own thread
__global__ void compute_pij_kernel(float *data, float *pij){

	int j, k;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float total_prob = 0;
       	float val, dist;


	if (i < dD){
		for (j = 0; j < dD; j++){
			dist = 0;
			if (i == j) pij[i*dD+j] = 0;
			else {
				for (k = 0; k < dP; k++) dist += (data[i*dP+k]-data[j*dP+k])*(data[i*dP+k]-data[j*dP+k]);
				val = expf((0.0-sqrt(dist))/(2.0*perp*perp));
				total_prob += val;
				pij[i*dD+j] = val;
			}
		}
		for (j = 0; j < dD; j++) pij[i*dD+j] /= total_prob; 
	}


}

//access function for compute_pij_kernel
void cuda_compute_pij(float *data, float *pij){

	dim3 threads(1024);
	dim3 blocks(ceil((float)dD / 1024.0));

	compute_pij_kernel<<<blocks,threads>>>(data, pij);

}

//balances pij for increased accuracy
__global__ void symettric_pij_kernel(float *pij){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float val;

	if (i < dD && j < dD && i < j){
	
		val = pij[i*dD+j]+pij[j*dD+i];
		val /= (2.0 * (float)dD);
		pij[i*dD+j] = val;
		pij[j*dD+i] = val;	
	
	}

}

//access function to symettric_pij_kernel
void cuda_symettric_pij(float *pij){

	int blockDims = ceil((float)dD / 32.0);

	dim3 threads(32,32);
	dim3 blocks(blockDims, blockDims);

	symettric_pij_kernel<<<blocks,threads>>>(pij);

}

__global__ void compute_qij_kernel(sol *sols, float *qij){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	float total, val, dist;

	if (i < dD){
		total = 0;
		for (j = 0; j < dD; j++){
			if (i == j) qij[i*dD+j] = 0;
			else {
				dist = (sols[i]->x - sols[j]->x)^2 + (sols[i]->y - sols[j]->y)^2;
				val = 1.0 / (1.0+sqrt(dist));
				total += val;
				qij_grid[i*dD+j] = val;	
			}
		}
		for (j = 0; j < dD; j++) qij[i*dD+j] /= total;
	}
}


}

//access function to compute_qij_kernel
void cuda_compute_qij(sol *sols, float *qij){

	dim3 threads(1024);
	dim3 blocks(ceil((float)dD / 1024.0));

	compute_qij_kernel<<<blocks, threads>>>(sols, qij);

}

__global__ void compute_gradients_kernel(sol *sols, float *pij, float *qij, sol *prev){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	float gradX, gradY, prevX, prevY, pq, ed;

	if (i < dD){
		gradX = 0; gradY = 0;
		for (j = 0; j < dD; j++){
			pq = (pij[i*dD+j] - qij[i*dD+j]);
			ed = (sols[i]->x - sols[j]->x)^2 + (sols[i]->y - sols[j]->y)^2;
			ed = 1.0 / (1.0 + sqrt(ed));
			gradX += pq * (sol[i]->x - sol[j]->x) * ed;
			gradY += pq * (sol[i]->y - sol[j]->y) * ed;
		}
		gradX *= 4; gradY *= 4;
		prevX = sols[i]->x; prevY = sols[i]->y;
		sols[i]->x += l_r*gradX + momemtum*(sols[i]->x - prev[i]->x);
		sols[i]->y += l_r*gradY + momemtum*(sols[i]->y - prev[i]->y);
		prev[i]->x = prevX; prev[i]->y = prevY;
	}

}

void cuda_compute_gradients(sol *sols, float *pij, float *qij, sol *prev){

	dim3 threads(1024);
	dim3 blocks(ceil((float)dD / 1024.0));

	compute_gradients_kernel<<<blocks, threads>>>(sols, pij, qij, prev);

}

void tsne_cuda(float *data, sol *sols, int iters){

	int t;
	cudaError_t cE;

	//same mallocs as baseline below
    	//float *pij_grid = (float *)malloc(dD*dD*sizeof(float));
    	//float *qij_grid = (float *)malloc(dD*dD*sizeof(float));
    	//sol *sols       = (sol *)  malloc(dD*sizeof(sol));
    	sol *prev_sols  = (sol *)  malloc(dD*sizeof(sol));
    	for (t = 0; t < dD; t++) prev_sols[t] = 0;
	random_solutions(sols, dD);

	//corresponding matrixes that live on device
	float *data_d, *pij_grid_d, *qij_grid_d;
	sol *sols_d, *prev_sols_d;

	cE = cudaMalloc((void **)&data_d, dD*dP*sizeof(float));
	if (cE != cudaSuccess){
		printf("Can not malloc memory on device\n");
		exit(1);
	}
	cudaMalloc((void **)&pij_grid_d,  dD*dD*sizeof(float));
	cudaMalloc((void **)&qij_grid_d,  dD*dD*sizeof(float));
	cudaMalloc((void **)&sols_d,      dD*sizeof(sol));
	cudaMalloc((void **)&prev_sols_d, dD*sizeof(sol));

	//put data_d onto device, make initial pij_grid on device, free data_d
	if (cudaMemcpy(data_d, data, dD*dP*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess){
		printf("Error copying data to device\n");
		exit(0);
	}

	cuda_compute_pij(data_d, pij_grid_d);
	
	//free the memory used for data, as it is no longer used
	if (cudaFree(data_d) != cudaSuccess){
		printf("Error freeing data on device\n");
		exit(0); 
	}

	cuda_symmetric_pij(pij_grid_d);

	cudaMemcpy(sols_d, sols, dD*sizeof(sol));
	cudaMemcpy(sols_prev_d, sols_prev, dD*sizeof(sol));
	for (t = 0; t < iters; t++){
		cuda_compute_qij(sols_d, qij_grid_d);
		cuda_compute_gradients(sols_d, pij_grid_d, qij_grid_d, sols_prev_d);
	}
	cudaMemcpy(&sols, sols_d, dD*sizeof(sols), cudaMemcpyDeviceToHost);
	
	cudaFree(pij_grid_d);
	cudaFree(qij_grid_d);
	cudaFree(sols_d);
	cudaFree(sols_prev_d);
	free(prev_sols);

}

//the main function
void tsne_baseline(int *data, sol *sols, int iters){
	
	int t;
	
	//malloc grids, init prev_sol
	float *pij_grid = (float *)malloc(dD*dD*sizeof(float));
	float *qij_grid = (float *)malloc(dD*dD*sizeof(float));
	//sol *sols       = (sol *)  malloc(dD*sizeof(sol));
	sol *prev_sols  = (sol *)  malloc(dD*sizeof(sol));
	for (t = 0; t < dD; t++) prev_sols[t] = 0;

	//prepare for loop
	compute_pij(data, pij_grid);
	symmetric_pij(pij_grid);
	random_solution(sols);

	//slowly move each x/y closer to it's true value
	for (t = 0; t<iters; t++){
		compute_qij(sols, qij_grid);
		compute_gradients(sols, pij_grid, qij_grid, prev_sols);
	}

	//free memory
	free(pij_grid);
	free(qij_grid);
	free(sols);
	free(prev_sols);
}


int main(int argc, void **argv){

	//get datafile name from argument
	//
	//open that file and parse data into 2D array
	//set globals, usually done with info about data and info for particular trial
	dD = 60000;
	dP = 784;
	perp = 25.0;
	l_r = 100; //look up Jacob’s 1988 ALR papr
	momemtum = 0.5; //change to 0.8 after 250 iters

	sol *sols = (sol *)malloc(dD*sizeof(sol));

	return 0;
}
