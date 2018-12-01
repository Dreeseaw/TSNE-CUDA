/* William Dreese + Steven Gschwind
 * t-SNE C baseline for mini project
 *
 * Proper import path before compilation:
 * 	export PATH=${PATH}:/usr/local/cuda-9.1/bin
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include <time.h>

typedef unsigned long long ull;

unsigned long long dD, dP;
float perp, l_r, momemtum;
int iters;
int *image_labels;
static FILE *FILE_POINTER = NULL;
static int EOF_FLAG = 0;

void setFilePointer(char *file_name);
float *getChunkedValues();
void closeFile();
int parseNextLine(float *data_points);
float getNextValue();

struct sol {
	float x,y;
};

void pf(int i){
	printf("here %d\n",i);
}

static double get_walltime() {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

void setFilePointer(char *file_name) {
    FILE_POINTER = fopen(file_name, "r");
    
    if (FILE_POINTER == NULL) {
        printf("Error opening file\n");
        exit(1);
    }
    EOF_FLAG = 0;
}

void getChunkedValues(float *data) {
    image_labels = (int *)malloc(dD * sizeof(int));

    // int throwaway = parseNextLine(data);   
 
    // Set image_labels and data values, image_labels will be used for cluster graph
    int count;
    for (count = 0; count < dD; count ++) {
        image_labels[count] = parseNextLine(&data[count * dP]);
    }
}

void closeFile() {
    fclose(FILE_POINTER);
    free(image_labels);
    FILE_POINTER = NULL;
}

int parseNextLine(float *data_points) {
    
    if (FILE_POINTER == NULL) {
        printf("No file found, call setFilePointer(char *file_name) before runnning\n");
        exit(1);
    }

    // Get label value for image creation
    int value = getNextValue();

    if (EOF_FLAG > 0) {
        return -1; // Finished parsing the file
    }
    
    int count = 0;
    while (count < dP) {
        data_points[count] = getNextValue();
        count += 1;
    }
    
    return value;
}

// Returns next value from csv file
float getNextValue() {
    char value[4] = {0,0,0,0}; // value is between 0 and 256
    int pos = 0;
    char current;
    
    while (1) {
        current = fgetc(FILE_POINTER);
        if (current == ',') {
            break;
        }
        else if (current == EOF) {
            EOF_FLAG = 1;
            break;
        }
        else if (current == '\n') {
            break;
        }
        value[pos] = current;
        pos += 1;
    }
    
    return atof(value);
}

void displayHelp() {
    printf("\nHELP\nThe program must be run in the format './tsne {filename.csv} {#rows} {#cols} {command} {command_input}....'\n");
    printf("The filename.csv, #rows and #cols inputs are necessary. \n#cols should be the number of data points, not including the label column\n");
    printf("\nBelow are a set of optional commands:\n");
    printf("\tCommand\t\tValue Type\tMeaning\t\n");
    printf("\t-----------------------------------------\n");
    printf("\t-perp\t\tfloat\t\tSet the perplexity value\n");
    printf("\t-learning_rate\tfloat\t\tSet the learning rate\n");
    printf("\t-momentum\tfloat\t\tSet the momentum\n");
    printf("\t-iters\t\tint\t\tSet the number of iterations\n\n");

    printf("For simplicities sake, the commands that we have used to run our test was:\n");
    printf("\t./tsne fashion-mnist_test.csv 10000 784\n");
    printf("We modified the optional input as necessary\n");
}

void parseCommandLineArguments(int argc, char **argv) {
    
    if (argc < 2) {
        printf("Illegal number of arguments supplied. Displaying help (-h)\n");
        displayHelp();
        exit(1);
    }
    
    if (strcmp(argv[1], "-h") == 0) {
        displayHelp();
        exit(0);
    } else {
        setFilePointer(argv[1]);
    }

    if (argc % 2 != 0 || argc < 4) {
        printf("Illegal number of arguments supplied. Displaying help (-h)\n");
        displayHelp();
        exit(1);
    }

    // Set columns and width
    dD = atoi(argv[2]);
    dP = atoi(argv[3]);
    
    // Set each of the optional commandline arguments
    int i;
    float arg_value;
    for (i = 4; i < argc; i += 2) {
        arg_value = atof(argv[i+1]);
        if (strcmp(argv[i], "-perp") == 0) {
            perp = arg_value;
        } else if (strcmp(argv[i], "-learning_rate") == 0) {
            l_r = arg_value;
        } else if (strcmp(argv[i], "-momentum") == 0) {
            momemtum = arg_value;
        } else if (strcmp(argv[i], "-iters") == 0) {
            iters = arg_value;
        } else {
            printf("Command (%s) not found\n", argv[i]);
        }
    }
    printf("Arguments:\n");
    printf("Perplexity: %g\n", perp);
    printf("Learning_rate: %g\n", l_r);
    printf("Momentum: %g\n", momemtum);
    printf("Iterations: %d\n", iters);
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

	unsigned long long i, j, k;
	float total_prob, val;


	for (i = 0; i < dD; i++){
		total_prob = 0;
		for (j = 0; j < dD; j++){
			val = 0;
			if (i == j) pij_grid[i*dD+j] = 0;
			else {							  
				for (k = 0; k < dP; k++) val += (data[i*dP+k]-data[j*dP+k])*(data[i*dP+k]-data[j*dP+k]);
				val = expf((0.0-sqrt(val))/(2.0*perp*perp));
				total_prob += val;
				pij_grid[i*dD+j] = val;
			}
		}
		for (j = 0; j < dD; j++) pij_grid[i*dD+j] /= total_prob; 
	}
}
/*
//compute true similarity scores between data points
void compute_pij_hbeta(float *data, float *pij_grid){

    unsigned long long i, j, k, tries;
    float total_prob, val, beta, bmax, bmin, dp, hval;
    float tol = 0.00001;
    float logp = logf(perp);

    float *dist_temp = (float *)malloc(dD*sizeof(float));

    for (i = 0; i < dD; i++){
        total_prob = 0;
        beta = 1.0;
        bmax = 10000.0;
        bmin = -10000.0;
        for (j = 0; j < dD; j++){
            val = 0;
            if (i == j) {
            pij_grid[i*dD+j] = 0;
            dist_temp[j] = 0.0;
        }
            else {
                for (k = 0; k < dP; k++) val += (data[i*dP+k]-data[j*dP+k])*(data[i*dP+k]-data[j*dP+k]);
                dist_temp[j] = sqrt(val);
            }
        }
        tries = 0;
        do {
            if (tries != 0){
                if ((hval-logp) > 0.0){
                    bmin = beta;
                    if (bmax == 10000 || bmax == -10000) beta *= 2.0;
                    else beta = (beta + bmax) / 2.0;
            }
            else {
                    bmax = beta;
                    if (bmin == 10000 || bmin == -10000) beta /= 2.0;
                    else beta = (beta + bmin) / 2.0;
                }
            }
            tries++;
            dp = 0;
            for (j = 0; j < dD; j++){
                val = expf((0.0-dist_temp[j])*beta);
                total_prob += val;
                pij_grid[i*dD+j] = val;
                dp += val*dist_temp[j];
            }
            hval = logf(total_prob) + ( beta * dp / total_prob);
            for (j = 0; j < dD; j++) pij_grid[i*dD+j] /= total_prob;
        } while (abs(hval-logp) > tol && tries < 50);
    }
    free(dist_temp);
}
*/

//pre-processing of pij  rid, saves time and increases visual accuracy
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
		sols[i].x = (float)rand() / (float)RAND_MAX;
		sols[i].y = (float)rand() / (float)RAND_MAX;
	}

}

//calculate euclidean distance between two 2D points
float sol_ed(sol i, sol j){

	return sqrt(((i.x-j.x)*(i.x-j.x))+((i.y-j.y)*(i.y-j.y)));

}

//calculate low-dimensionality similarity grid
void compute_qij(sol *sols, float *qij_grid){

	unsigned long long i, j;
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
		// TODO: subtract so k!=i
	}
}

//calculates and applies gradients to each solution
void compute_gradients(sol *sols, float *pij_grid, float *qij_grid, sol *prev_sols){

	ull i, j;
	float gradX, gradY, prevX, prevY, pq, ed;

	for (i = 0; i < dD; i++){
		gradX = 0; gradY = 0;
		for (j = 0; j < dD; j++){
			pq = (pij_grid[i*dD+j] - qij_grid[i*dD+j]);
			ed = 1.0 / (1.0 + sol_ed(sols[i],sols[j]));
			gradX += pq * (sols[i].x - sols[j].x) * ed;
			gradY += pq * (sols[i].y - sols[j].y) * ed;
		}
		gradX *= 4; gradY *= 4;
		prevX = sols[i].x; prevY = sols[i].y;
		sols[i].x += l_r*gradX + momemtum*(sols[i].x - prev_sols[i].x);
		sols[i].y += l_r*gradY + momemtum*(sols[i].y - prev_sols[i].y);
		prev_sols[i].x = prevX; prev_sols[i].y = prevY;
		}

}

//honestly, the exact same as baseline compute_pij, except each row is sent to it's own thread
__global__ void compute_pij_kernel_A(float *data, float *pij, ull dD, ull dP, float perp, ull offset){

	ull k;
	ull i = (ull) (blockIdx.x * blockDim.x + threadIdx.x);
	ull j = (ull) (offset * blockDim.y + threadIdx.y);	

       	float dist = 0;
	float val;

	if (i < dD && j < dD){
		for (k = 0; k < dP; k++) dist += (data[i*dP+k]-data[j*dP+k])*(data[i*dP+k]-data[j*dP+k]);
		val = expf((0.0-sqrt(dist))/(2.0*perp*perp));
		pij[i*dD+j] = val;
		if (i == j) pij[i*dD+j] = 0.0;
	}
	

}

//compute total per row, divide each element
__global__ void compute_pij_kernel_B(float *pij, ull dD){

	ull i = (ull) (blockIdx.x * blockDim.x + threadIdx.x);
	ull j;

	float total = 0;

	if (i < dD){
		for (j = 0; j < dD; j++) total += pij[i*dD+j];
		for (j = 0; j < dD; j++) pij[i*dD+j] /= total;
	}
}

//balances pij for increased accuracy
__global__ void symettric_pij_kernel(float *pij, ull dD){

        ull i = (ull) blockIdx.x * blockDim.x + threadIdx.x;
        ull j = (ull) blockIdx.y * blockDim.y + threadIdx.y;

        float val;

        if (i < dD && j < dD && i < j){

                val = (pij[i*dD+j]+pij[j*dD+i]) / (float)(2*dD);
                pij[i*dD+j] = val;
                pij[j*dD+i] = val;

        }

}

//access function for compute_pij_kernel
void cuda_compute_pij(float *data, float *pij){

	int ys = ceil((float)dD / 32.0);

	dim3 threads1D(1024);
	dim3 blocks1D(ceil((float)dD / 1024.0));
	dim3 threads2D(32,32);
	dim3 blocks2D(ys, 1);
	dim3 blocks2Da(ys, ys);

	for (ull y = 0; y < ys; y++){
		compute_pij_kernel_A<<<blocks2D, threads2D>>>(data, pij, dD, dP, perp, y);
	}
	
	cudaFree(data);
	
	compute_pij_kernel_B<<<blocks1D, threads1D>>>(pij, dD);
	symettric_pij_kernel<<<blocks2Da, threads2D>>>(pij, dD);

}

__global__ void compute_qij_kernel(sol *sols, float *qij, ull dD){
	
	ull i = (ull) blockIdx.x * blockDim.x + threadIdx.x;
	ull j;
	float total, val, dist;

	if (i < dD){
		total = 0;
		for (j = 0; j < dD; j++){
			if (i == j) qij[i*dD+j] = 0;
			else {
				dist = (sols[i].x - sols[j].x)*(sols[i].x-sols[j].x) + (sols[i].y - sols[j].y)*(sols[i].y-sols[j].y);
				val = 1.0 / (1.0+sqrt(dist));
				total += val;
				qij[i*dD+j] = val;	
			}
		}
		for (j = 0; j < dD; j++) qij[i*dD+j] /= total;
	}
}


//access function to compute_qij_kernel
void cuda_compute_qij(sol *sols, float *qij){

	dim3 threads(1024);
	dim3 blocks(ceil((float)dD / 1024.0));

	compute_qij_kernel<<<blocks, threads>>>(sols, qij, dD);

}

__global__ void compute_gradients_kernel(sol *sols, float *pij, float *qij, sol *prev, ull dD, float momemtum, float l_r){

	ull i = (ull) (blockIdx.x * blockDim.x + threadIdx.x);
	ull j;
	float gradX, gradY, prevX, prevY, pq, ed;

	if (i < dD){
		gradX = 0; gradY = 0;
		for (j = 0; j < dD; j++){
			pq = (pij[i*dD+j] - qij[i*dD+j]);
			ed = (sols[i].x - sols[j].x)*(sols[i].x-sols[j].x) + (sols[i].y - sols[j].y)*(sols[i].y-sols[j].y);
			ed = 1.0 / (1.0 + sqrt(ed));
			gradX += pq * (sols[i].x - sols[j].x) * ed;
			gradY += pq * (sols[i].y - sols[j].y) * ed;
		}
		gradX *= 4; gradY *= 4;
		prevX = sols[i].x; prevY = sols[i].y;
		sols[i].x += l_r*gradX + momemtum*(sols[i].x - prev[i].x);
		sols[i].y += l_r*gradY + momemtum*(sols[i].y - prev[i].y);
		prev[i].x = prevX; prev[i].y = prevY;
	}

}

void cuda_compute_gradients(sol *sols, float *pij, float *qij, sol *prev){

	dim3 threads(1024);
	dim3 blocks(ceil((float)dD / 1024.0));

	compute_gradients_kernel<<<blocks, threads>>>(sols, pij, qij, prev, dD, momemtum, l_r);
	cudaThreadSynchronize();
}

void cEcheck(cudaError_t cE, const char *type){
	if (cE != cudaSuccess){
		printf("Error while %s memory.\n",type);
		printf( cudaGetErrorString( cudaGetLastError()));
		exit(1);
	}	
}

void tsne_cuda(float *data, sol *sols, int iters){

	int t;

	//same mallocs as baseline below
    	sol *prev_sols  = (sol *)  malloc(dD*sizeof(sol));
    	for (t = 0; t < dD; t++){ prev_sols[t].x = 0.0; prev_sols[t].y = 0.0; }
	random_solutions(sols);

	//corresponding matrixes that live on device
	float *data_d, *pij_grid_d, *qij_grid_d;
	sol *sols_d, *prev_sols_d;

	cEcheck( cudaMalloc((void **)&data_d,      dD*dP*sizeof(float)), "allocating" );
	cEcheck( cudaMalloc((void **)&pij_grid_d,  dD*dD*sizeof(float)), "allocating" );

	cEcheck( cudaMemcpy(data_d, data, dD*dP*sizeof(float), cudaMemcpyHostToDevice), "transferring" );

	cuda_compute_pij(data_d, pij_grid_d); //also frees data

	printf("Making pij (CUDA)\n");

	cEcheck( cudaMalloc((void **)&qij_grid_d,  dD*dD*sizeof(float)), "allocating" );
	cEcheck( cudaMalloc((void **)&sols_d,      dD*sizeof(sol)),      "allocating" );
	cEcheck( cudaMalloc((void **)&prev_sols_d, dD*sizeof(sol)),      "allocating" );
	
	cEcheck( cudaMemcpy(sols_d,      sols,      dD*sizeof(sol), cudaMemcpyHostToDevice), "transferring" );
	cEcheck( cudaMemcpy(prev_sols_d, prev_sols, dD*sizeof(sol), cudaMemcpyHostToDevice), "transferring" );;

	for (t = 0; t < iters; t++){
		cuda_compute_qij(sols_d, qij_grid_d);
		cuda_compute_gradients(sols_d, pij_grid_d, qij_grid_d, prev_sols_d);
		if (t == 250) momemtum = 0.8;
	}

	cEcheck( cudaMemcpy(sols, sols_d, dD*sizeof(sol), cudaMemcpyDeviceToHost), "transferring" );

	pf(66);

	cudaFree(pij_grid_d);
	cudaFree(qij_grid_d);
	cudaFree(sols_d);
	cudaFree(prev_sols_d);

	free(prev_sols);

}

//the main function
void tsne_baseline(float *data, sol *sols, int iters){
	
	int t;
	
	//malloc grids, init prev_sol
	float *pij_grid = (float *)malloc(dD*dD*sizeof(float));
	float *qij_grid = (float *)malloc(dD*dD*sizeof(float));
	
	sol *prev_sols  = (sol *)  malloc(dD*sizeof(sol));
	for (t = 0; t < dD; t++){
		prev_sols[t].x = 0.0;
		prev_sols[t].y = 0.0;
	}

	//prepare for loop
	compute_pij(data, pij_grid);
	symmetric_pij(pij_grid);
	random_solutions(sols);
	
	//slowly move each x/y closer to it's true value
	for (t = 0; t<iters; t++){
		compute_qij(sols, qij_grid);
		compute_gradients(sols, pij_grid, qij_grid, prev_sols);
	}

	//free memory
	free(pij_grid);
	free(qij_grid);
	free(prev_sols);
}

void normalizeSols(sol *solArray, int data_length) {
	// Normalize x values;
	float min, max;
	min = solArray[0].x;
	max = solArray[0].x;
	for (int i = 0; i < data_length; i++) {
		min = min > solArray[i].x ? solArray[i].x : min;
		max = max < solArray[i].x ? solArray[i].x : max;
	}

	float diff = max - min;	
	for(int i = 0; i < data_length; i++) {
		solArray[i].x = (solArray[i].x - min) / diff;
	}
	
	// Normalize y values;
	min = solArray[0].y;
	max = solArray[0].y;	
	for (int i = 0; i < data_length; i++) {
		min = min > solArray[i].y ? solArray[i].y : min;
		max = max < solArray[i].y ? solArray[i].y : max;
	}
	diff = max - min;
	for(int i = 0; i < data_length; i++) {
		solArray[i].y = (solArray[i].y - min) / diff;
	}	
}

void outputSols(sol *solArray, int data_length) {
	FILE *fp = fopen("output.txt", "w");
	for (int t = 0; t < data_length; t++) fprintf(fp,"%d, %f, %f\n",image_labels[t], solArray[t].x, solArray[t].y);
	fclose(fp);
}


int main(int argc, char **argv){

	if (argc == 1) {
        	printf("File name not supplied as argument, quitting\n");
        	exit(1);
    	}
	
	cudaDeviceProp pp;
	cudaGetDeviceProperties(&pp, 0);
	//printf("%zu\n",pp.warpSize);

	//default values		
	dD = 10000;
	dP = 784;
	perp = 25.0;
	l_r = 100.0; //look up Jacob’s 1988 ALR papr
	momemtum = 0.5; //change to 0.8 after 250 iters
	iters = 1000;

	parseCommandLineArguments(argc, argv);

	sol *sols = (sol *)malloc(dD*sizeof(sol));
	sol *solsCUDA = (sol *)malloc(dD*sizeof(sol));

	float *data = (float *)malloc(dD*dP*sizeof(float));
	getChunkedValues(data);

	double ss, se, ce;
	ss = get_walltime();
	tsne_baseline(data, sols, iters);
	se = get_walltime();
	tsne_cuda(data, solsCUDA, iters);
	ce = get_walltime();	
	
	printf("times: \n\tbaseline: %f\n\tcuda: %f\n",se-ss,ce-se);
	
	normalizeSols(solsCUDA, dD);
	outputSols(solsCUDA, dD);

	closeFile();	
	free(data);
	free(sols);
	free(solsCUDA);
	
	return 0;
}
