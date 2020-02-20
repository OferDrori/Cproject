#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define WTYPE 0
#define FILE_NAME "C:\\Users\\cudauser\\Downloads\\data1.txt"// put here the file location
//Ofer Drori 313264541 un parallel

void readDataFromFile(int* N, int* K, double* dT, double* tMax, double*alpha, int* LIMIT, double*QC,
	double** pointsLocation,  double** pointsVvectors, int** pointsSign, int** pointsSignSet);
void CalcNewPointPosition(double* pointsLocation, double* velocities,
	int N, int K, double dt);
void setWType(double* w, int K);
void initPointsSigns(int n, int k, double * pointsLocation, int* pointsSignSet, double* w);
void updateWValues(double* w, double* pointLocation, int sign, int K, double alpha);
int checkBadPointIndex(int* pointsSigns, int* pointsSignSet, int N, int lastBadIndex);
double getQuailty(int* pointsSign, int* pointsSignSet, int n);
void printResult(double* w, int k, double time);
void initPointsVariables(double** pointsLocation, double** pointsVvectors, int** pointsSignSet, int** pointsSign, int K, int N);
void freePointsVariables(double**pointsLocation, double**pointsVvectors, int**pointsSign, int**pointsSignSet, double**w);
	int main(int argc, char* argv[])
{
	

	clock_t  start, end;
	double cpu_time_used;
	start = clock();
	int N, K, LIMIT, iterationCount = 0;
	int allPointsOk = 0, indexOfBadPoint = 0, lastBadIndex = 0;
	double dT, alpha, QC, tMax, q , time = 0;
	double* pointsLocation, *pointsVvectors;
	int* pointsSign, *pointsSignSet;
	
	readDataFromFile(&N, &K, &dT, &tMax, &alpha, &LIMIT, &QC, &pointsLocation,
		 &pointsVvectors, &pointsSign, &pointsSignSet);


	double* w = (double*)malloc((K + 1) * sizeof(double));


	q = QC + 1;
	while (time < tMax && q > QC)
	{
			lastBadIndex = 0;
			allPointsOk = 0;
			iterationCount = 0;
			setWType(w, K);// set all w values to 0 in the start
		while (iterationCount < LIMIT && allPointsOk == 0)
		{
			
			initPointsSigns(N, K, pointsLocation , pointsSignSet, w);// pointsSignSet is the array that update every routine
			                                                         //this func we change him to the right sign according to the w value and pointsLocation. 

			 
			
			indexOfBadPoint = checkBadPointIndex(pointsSign, pointsSignSet, N, lastBadIndex);// pointsSign is a permanent value that save the sing of every point
		                                                                                     //In this func we check the index that not match between pointsSignSet and pointsSign that appear after the last bad index .	 
			
			if (indexOfBadPoint == -1)// if we dont found any problamatic index checkBadPointIndex return -1.
			{
				allPointsOk = 1;
			}
			else
			{
				
				updateWValues(w, pointsLocation + (indexOfBadPoint * K), pointsSignSet[indexOfBadPoint], K, alpha);// update w value according to the last bad point index.
		
				allPointsOk = 0;
				lastBadIndex = indexOfBadPoint;
			}
			iterationCount++;
		}
		
		q = getQuailty(pointsSign, pointsSignSet, N);
		if (q > QC)
		{
			time += dT;// update the time
			CalcNewPointPosition(pointsLocation, pointsVvectors, N, K, dT);// chage the pointsLocation according to the pointsVvectors and the dT 
		}
	}
	
		printResult(w, K+1, time);
		end = clock();
		freePointsVariables(&pointsLocation,&pointsVvectors, &pointsSign, &pointsSignSet,&w);
		cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
		printf("the time of unparallel program is   %lf \n ", cpu_time_used);
}
	

void freePointsVariables(double**pointsLocation, double**pointsVvectors, int**pointsSign, int**pointsSignSet,double**w) 
{
		free(*pointsLocation);
		free(*pointsVvectors);
		free(*pointsSign);
		free(*pointsSignSet);
		free(*w);
}

	


int checkBadPointIndex(int* pointsSigns, int* pointsSignSet, int N,  int lastBadIndex)
{
	int i = lastBadIndex;
	while (i < N)
	{
		if (pointsSigns[i] != pointsSignSet[i])
		{

			return i;
		}
		i++;
	}

	i = 0;
	while (i < lastBadIndex)
	{
		if (pointsSigns[i] != pointsSignSet[i])
		{
			return i;
		}
		i++;
	}
	return -1;
}


void readDataFromFile(int* N, int* K, double* dT, double* tMax, double*alpha, int* LIMIT, double*QC,
	double** pointsLocation, double** pointsVvectors, int** pointsSign, int** pointsSignSet)
{
	FILE* f = fopen(FILE_NAME, "r");
	if (f == NULL)
	{
		perror("Error");
		printf("Could not open file %s", FILE_NAME);
	}
	fscanf(f, "%d %d %lf %lf %lf %d %lf\n", N, K, dT, tMax, alpha, LIMIT, QC);
	initPointsVariables(pointsLocation, pointsVvectors, pointsSignSet, pointsSign, *K, *N);
	int indexL = 0;
	int indexV = 0;// for convenient 
	for (int i = 0; i < *N; i++)
	{
		for (int j = 0; j < *K; j++)
		{
			fscanf(f, "%lf", &((*pointsLocation)[indexL]));
			indexL++;
		}
		for (int j = 0; j < *K; j++)
		{
			fscanf(f, "%lf", &((*pointsVvectors)[indexV]));
			indexV++;
		}
		fscanf(f, "%d\n", &((*pointsSign)[i]));
	}
	fclose(f);
}


void initPointsVariables(double** pointsLocation, double** pointsVvectors, int** pointsSignSet, int** pointsSign, int K, int N)
{
	*pointsLocation = (double*)malloc(sizeof(double)*N*K);
	*pointsVvectors = (double*)malloc(sizeof(double)*N*K);
	*pointsSign = (int*)malloc(sizeof(int)*N);
	*pointsSignSet = (int*)calloc(N, sizeof(int));
}

void initPointsSigns(int n, int k, double * pointsLocation,int* pointsSignSet, double* w)
{
	int index = 0;
	for (int i = 0; i < n; i++)
	{
		double result = w[k];
		for (int j = 0; j < k; j++)
		{
			result += w[j] * pointsLocation[index];
			index++;
		}
		if (result >= 0)
			pointsSignSet[i] = 1;
		else {
			pointsSignSet[i] = -1;
		}

	
	}
}

void CalcNewPointPosition(double* pointsLocation, double* velocities,
	int N, int K, double dt)
{
	int index = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K; j++)
		{
			pointsLocation[index] = pointsLocation[index] + velocities[index] * dt;
			index++;
		}
	}
}

double getQuailty(int* pointsSign, int* pointsSignSet, int n)
{
	int numOfMisses = 0;
	int i;
	for (i = 0; i < n; i++)
	{
		if (pointsSign[i] != pointsSignSet[i])
			numOfMisses++;
	}
	return (double)numOfMisses / n;
}
void setWType(double* w, int K)
{
	for (int i = 0; i <= K; i++)
		w[i] = 0;
}
void updateWValues(double* w, double* pointLocation, int sign, int K, double alpha)
{
	w[K] += (-sign)*alpha;
	for (int i = 0; i < K; i++)
	{
		w[i] += -(sign)* alpha * pointLocation[i];
	}
}
void printResult(double* w, int k, double time)
{
	int i = 0;
	printf("the W that match the points is\n");
	for (i = 0; i < k; i++)
	{
		printf("%lf \n", w[i]);

	}
	printf("in time %lf \n", time);



}
