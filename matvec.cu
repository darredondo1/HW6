#include <cuda.h>
#include "timer.h"
#include <math.h>

__global__ void
matVecMultKernel(float* A, float* B, float* C, int n)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (row < n)
    {
        float val = 0;
        for (int i=0; i<n; i++)
        {
            val += A[i*n + row] * B[row];
        }
        C[row] = val;
    }
}

void matVecMult(float* A, float* B, float* C, int n)
{
    float val;
    for (int i=0; i<n; i++)
    {
        val = 0;
        for (int j=0; j<n; j++)
        {
            val += A[i*n+j] * B[j];
        }
        C[i] = val;
    }
}

void matrixMult(float* A, float* B, float* C, int n)
{
    float val;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            val = 0;
            for (int k = 0; k < n; k++)
                val += A[i*n+k] * B[k*n+j];
            C[i*n+j] = val;
        }
    }
}

double sum(float* C, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            s += C[i*n+j];
    return s;
}

double vecSum(float* C, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        s += C[i];
    return s;
}

int main(int argc, char* argv[])
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    double t0, tfinal;

    int n = atoi(argv[1]);
    int size = n*n*sizeof(float);
    int vecsize=n*sizeof(float);
    int numTests = atoi(argv[2]);

///
    //make file
    FILE * fPtr;
    char fPath[100];
    sprintf(fPath,"Problem2/matvec_N_%d.txt",n);
///DA

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(vecsize);
    h_C = (float*)malloc(vecsize);
    
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, vecsize);
    cudaMalloc((void**)&d_C, vecsize);


   for (int test=0;test<numTests;test++)
   {
        for (int i = 0; i < n*n; i++)
        {
            h_A[i] = (float) rand();
            if (i<n) h_B[i] = (float) rand();
        }
        
        t0 = get_time();
        matVecMult(h_A, h_B, h_C, n);
        tfinal = get_time() - t0;
        printf("MatVecMult Time %e, Sum %e\n", tfinal, vecSum(h_C, n));

        t0 = get_time();
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, vecsize, cudaMemcpyHostToDevice);
        matVecMultKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
        cudaMemcpy(h_C, d_C, vecsize, cudaMemcpyDeviceToHost);
        tfinal = get_time() - t0;
        printf("MatVecMultKernel Time %e, Sum %e\n", tfinal, vecSum(h_C, n));

    ///
        //save time
        fPtr = fopen(fPath ,"a");
        if (fPtr == NULL) exit(EXIT_FAILURE);
        fprintf(fPtr,"%e\n",tfinal);
        fclose(fPtr);
    ///DA

        
   }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}

