#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define WTYPE 0
#define FILE_NAME "C:\\Users\\cudauser\\Downloads\\data1.txt"
//Ofer Drori 313264541 un parallel

void readDataFromFileUnParallel(int* N, int* K, double* dT, double* tMax, double*alpha, int* LIMIT, double*QC,
	double** pointsLocation, double** pointsVvectors, int** pointsSign, int** pointsSignSet);
void CalcNewPointPositionUnParallel(double* pointsLocation, double* velocities,
	int N, int K, double dt);
void setWTypeUnParallel(double* w, int K);
void initPointsSignsUnParallel(int n, int k, double * pointsLocation, int* pointsSignSet, double* w);
void updateWValuesUnParallel(double* w, double* pointLocation, int sign, int K, double alpha);
int checkBadPointIndexUnParallel(int* pointsSigns, int* pointsSignSet, int N, int lastBadIndex);
double getQuailtyUnParallel(int* pointsSign, int* pointsSignSet, int n);
void printResultUnParallel(double* w, int k, double time);
void initPointsVariablesUnParallel(double** pointsLocation, double** pointsVvectors, int** pointsSignSet, int** pointsSign, int K, int N);
void freePointsVariablesUnParallel(double**pointsLocation, double**pointsVvectors, int**pointsSign, int**pointsSignSet);
void unparallel();

void unparallel()
{


	clock_t  start, end;
	double cpu_time_used;
	start = clock();
	int N, K, LIMIT, iterationCount = 0;
	int allPointsOk = 0, indexOfBadPoint = 0, lastBadIndex = 0;
	double dT, alpha, QC, tMax, q, time = 0;
	double* pointsLocation, *pointsVvectors;
	int* pointsSign, *pointsSignSet;

	readDataFromFileUnParallel(&N, &K, &dT, &tMax, &alpha, &LIMIT, &QC, &pointsLocation,
		&pointsVvectors, &pointsSign, &pointsSignSet);


	double* w = (double*)malloc((K + 1) * sizeof(double));


	q = QC + 1;
	while (time < tMax && q > QC)
	{
		lastBadIndex = 0;
		allPointsOk = 0;
		iterationCount = 0;
		setWTypeUnParallel(w, K);// set all w values to 0 in the start
		while (iterationCount < LIMIT && allPointsOk == 0)
		{

			initPointsSignsUnParallel(N, K, pointsLocation, pointsSignSet, w);// pointsSignSet is the array that update every routine
																	//this func we change him to the right sign according to the w value and pointsLocation. 



			indexOfBadPoint = checkBadPointIndexUnParallel(pointsSign, pointsSignSet, N, lastBadIndex);// pointsSign is a permanent value that save the sing of every point
																							 //In this func we check the index that not match between pointsSignSet and pointsSign that appear after the last bad index .	 

			if (indexOfBadPoint == -1)// if we dont found any problamatic index checkBadPointIndex return -1.
			{
				allPointsOk = 1;
			}
			else
			{

				updateWValuesUnParallel(w, pointsLocation + (indexOfBadPoint * K), pointsSignSet[indexOfBadPoint], K, alpha);// update w value according to the last bad point index.

				allPointsOk = 0;
				lastBadIndex = indexOfBadPoint;
			}
			iterationCount++;
		}

		q = getQuailtyUnParallel(pointsSign, pointsSignSet, N);
		if (q > QC)
		{
			time += dT;// update the time
			CalcNewPointPositionUnParallel(pointsLocation, pointsVvectors, N, K, dT);// chage the pointsLocation according to the pointsVvectors and the dT 
		}
	}

	printResultUnParallel(w, K + 1, time);
	end = clock();
	freePointsVariablesUnParallel(&pointsLocation, &pointsVvectors, &pointsSign, &pointsSignSet);
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("the time of unparallel program is   %lf \n ", cpu_time_used);
}


void freePointsVariablesUnParallel(double**pointsLocation, double**pointsVvectors, int**pointsSign, int**pointsSignSet)
{
	free(*pointsLocation);
	free(*pointsVvectors);
	free(*pointsSign);
	free(*pointsSignSet);
}




int checkBadPointIndexUnParallel(int* pointsSigns, int* pointsSignSet, int N, int lastBadIndex)
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


void readDataFromFileUnParallel(int* N, int* K, double* dT, double* tMax, double*alpha, int* LIMIT, double*QC,
	double** pointsLocation, double** pointsVvectors, int** pointsSign, int** pointsSignSet)
{
	FILE* f = fopen(FILE_NAME, "r");
	if (f == NULL)
	{
		perror("Error");
		printf("Could not open file %s", FILE_NAME);
	}
	fscanf(f, "%d %d %lf %lf %lf %d %lf\n", N, K, dT, tMax, alpha, LIMIT, QC);
	initPointsVariablesUnParallel(pointsLocation, pointsVvectors, pointsSignSet, pointsSign, *K, *N);
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


void initPointsVariablesUnParallel(double** pointsLocation, double** pointsVvectors, int** pointsSignSet, int** pointsSign, int K, int N)
{
	*pointsLocation = (double*)malloc(sizeof(double)*N*K);
	*pointsVvectors = (double*)malloc(sizeof(double)*N*K);
	*pointsSign = (int*)malloc(sizeof(int)*N);
	*pointsSignSet = (int*)calloc(N, sizeof(int));
}

void initPointsSignsUnParallel(int n, int k, double * pointsLocation, int* pointsSignSet, double* w)
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

void CalcNewPointPositionUnParallel(double* pointsLocation, double* velocities,
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

double getQuailtyUnParallel(int* pointsSign, int* pointsSignSet, int n)
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
void setWTypeUnParallel(double* w, int K)
{
	for (int i = 0; i <= K; i++)
		w[i] = 0;
}
void updateWValuesUnParallel(double* w, double* pointLocation, int sign, int K, double alpha)
{
	w[K] += (-sign)*alpha;
	for (int i = 0; i < K; i++)
	{
		w[i] += -(sign)* alpha * pointLocation[i];
	}
}
void printResultUnParallel(double* w, int k, double time)
{
	int i = 0;
	printf("the W that match the points is\n");
	for (i = 0; i < k; i++)
	{
		printf("%lf \n", w[i]);

	}
	printf("in time %lf \n", time);



}
