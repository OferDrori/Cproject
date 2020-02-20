#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma once
#define ROOT 0
#define WTYPE 0
#define DATA_TAG 1
#define FILE_NAME "C:\\Users\\cudauser\\Downloads\\data1.txt"
#define OUTPUT_FILE "C:\\Users\\cudauser\\Downloads\\output.txt"

//Ofer Drori 313264541 parallel
void freePointsVariables(double**pointsLocation, double**pointsLocationPermanent, double**pointsVvectors, int**pointsSign, int**pointsSignSet, double** w);
void readDataFromFile(int* N, int* K, double* dT, double* tMax, double*alpha, int* LIMIT, double*QC,
	double** pointsLocation, double** pointsVvectors, int** pointsSign, int** pointsSignSet);
void setWType(double* w, int K);
int checkBadPointIndex(int* pointsSigns, int* pointsSignSet, int N, int lastBadIndex);
int checkIteration(int* okIterationForTime, int numprocs);
void initPointsVariables(double** pointsLocation, double** points_velocities, int** points_signs, int** points_sets, int K, int N);
cudaError_t calculatePointsCuda(double* points, int N, int K, double * w, int *results);
void CalcNewPointPosition(double* pointsLocationPernament, double* pointsLocation, double* velocities,
	int N, int K, double time);
void updateWValues(double* w, double* pointValues, int sign, int K, double alpha);
double getQuailtyWithOpenMP(int* pointsSign, int* pointsSignSet, int n);
void printResult(double* w, int k, double time, double q);
void copyarray(double * pointsLocation, double * pointsLocationPernament, int size);
void printResultTofile(double* w, int k, double time, double q);

int main(int argc, char* argv[])
{
	int myid, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
		clock_t  start, end;
		double cpu_time_used;
		start = clock();
		int N, K, LIMIT, iterationCount = 0;
		int allPointsOk = 0, indexOfBadPoint = 0, lastBadIndex = 0;
		double dT, alpha, QC, tMax;
		double* pointsLocation, *pointsVvectors;
		int* pointsSign, *pointsSignSet;
		int	indexOkTime = -1;
		MPI_Status status;
		int * okIterationForTime;
		double time;
		if (myid == ROOT)
		{
			readDataFromFile(&N, &K, &dT, &tMax, &alpha, &LIMIT, &QC, &pointsLocation,
				&pointsVvectors, &pointsSign, &pointsSignSet);
			okIterationForTime = (int*)malloc(numprocs * sizeof(int));

		}
		MPI_Bcast(&K, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(&N, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(&LIMIT, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(&dT, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(&alpha, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(&QC, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(&tMax, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

		if (myid != ROOT) {
			initPointsVariables(&pointsLocation, &pointsVvectors, &pointsSignSet, &pointsSign, K, N);
		}

		MPI_Bcast(pointsLocation, N*K, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(pointsVvectors, N*K, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(pointsSign, N, MPI_INT, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(pointsSignSet, N, MPI_INT, ROOT, MPI_COMM_WORLD);
		//The logic on the parrallel sulution is that all the proces knows on all the points vairables in the begining and 
		// every proces culculate his on solution in deferrent time. in the end the ROOT peek the one that work in the minimum time.
		// if there is no optimal solution, all the proces continue to calculate in diffrent time.

		double q = QC + 1;
		double* pointsLocationPermanent = (double*)malloc((K*N) * sizeof(double));
		copyarray(pointsLocation, pointsLocationPermanent, N*K);
		double* w = (double*)malloc((K + 1) * sizeof(double));
		int myIterationIsOk = 0;
		double i;
		for (i = myid*dT; i < tMax; i + numprocs)// this for loop every procec get his specific time 
		{
			CalcNewPointPosition(pointsLocationPermanent, pointsLocation, pointsVvectors, N, K, i);// this func change pointsLocation according to the time
			setWType(w, K);// func that set w values to 0
			allPointsOk = 0;
			lastBadIndex = 0;
			iterationCount = 0;

			while (iterationCount < LIMIT && allPointsOk == 0)
			{
				// CALL TO CUDA
				calculatePointsCuda(pointsLocation, N, K, w, pointsSignSet);//// pointsSignSet is the array that update every routine
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

			q = getQuailtyWithOpenMP(pointsSign, pointsSignSet, N);

			if (q < QC)
			{

				myIterationIsOk = 1;
			}

			// every procec have a variable myIterationIsOk that said if his Iteration is ok 
			MPI_Gather(&myIterationIsOk, 1, MPI_INT, okIterationForTime, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			// now the ROOT have a array with 0 and 1 that says in this spesefic time the iteration was good 
			// the array is ordered  by time the first index has the minimum time and continuity 
			if (myid == ROOT)
			{
				indexOkTime = checkIteration(okIterationForTime, numprocs);// the ROOT check the right index 
			}
			MPI_Bcast(&indexOkTime, 1, MPI_INT, ROOT, MPI_COMM_WORLD);// all the procec knows which procec get the right iteration(consider by time )

			if (myid == indexOkTime)
			{
				if (indexOkTime == ROOT)
				{
					//printResult(w, K, time, q);// for testing print to the screen 
					printResultTofile(w, K, time, q);
				}
				else {


					MPI_Send(w, K + 1, MPI_DOUBLE, ROOT, 2, MPI_COMM_WORLD);// just the ROOT is printing to the screen
					MPI_Send(&q, 1, MPI_DOUBLE, ROOT, 3, MPI_COMM_WORLD);// so we need to send him all the print info
					MPI_Send(&i, 1, MPI_DOUBLE, ROOT, 4, MPI_COMM_WORLD);


				}
				break;
			}
			else
			{
				if (indexOkTime != -1)// if indexOkTime=-1  its mean that in this iteration thare is no corect 
				                    	//option and we need to check in the diffrent time
				{
					if (myid == ROOT)// if I am ROOT and there was right answer from difrrent proces 
					{

						MPI_Recv(w, K + 1, MPI_DOUBLE, indexOkTime, 2, MPI_COMM_WORLD, &status);
						MPI_Recv(&q, 1, MPI_DOUBLE, indexOkTime, 3, MPI_COMM_WORLD, &status);
						MPI_Recv(&time, 1, MPI_DOUBLE, indexOkTime, 4, MPI_COMM_WORLD, &status);
						//printResult(w, K, time, q);// for testing print to the screen 
					     printResultTofile(w, K, time, q);
						free(okIterationForTime);
					}
					break;
				}

			}

		}
		freePointsVariables(&pointsLocation, &pointsLocationPermanent, &pointsVvectors, &pointsSign, &pointsSignSet,&w);
		if (myid == ROOT)
		{
			end = clock();
			cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
			printf("the time of parallel program is   %lf \n ", cpu_time_used);
		}
		MPI_Finalize();
	}


void freePointsVariables(double**pointsLocation, double**pointsLocationPermanent, double**pointsVvectors, int**pointsSign, int**pointsSignSet, double** w)
{
		free(*pointsLocationPermanent);
		free(*pointsLocation);
		free(*pointsVvectors);
		free(*pointsSign);
		free(*pointsSignSet);
}
	int checkIteration(int* okIterationForTime, int numprocs)
	{
		for (int i = 0; i < numprocs; i++)
		{
			if (okIterationForTime[i] == 1)
				return i;
		}
		return -1;
	}

int checkBadPointIndex(int* pointsSigns, int* pointsSignSet, int N, int lastBadIndex) {
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
		int indexV = 0;
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



void initPointsVariables(double** pointsLocation, double** points_velocities, int** points_signs, int** points_sets, int K, int N)
{
	*pointsLocation = (double*)malloc(sizeof(double)*N*K);
	*points_velocities = (double*)malloc(sizeof(double)*N*K);
	*points_signs = (int*)malloc(sizeof(int)*N);
	*points_sets = (int*)malloc(sizeof(int)*N);
}

void culculatePointsSigns(int N, int K, int* pointsSign, int *pointsLocation, double* w)
{
	for (int i = 0; i < N; i++)
	{
		double result = w[K];
		for (int j = 0; j < K; j++)
		{
			result += w[j] * pointsLocation[j];
		}
		if (result >= 0)
			pointsSign[i] = 1;
		else {
			pointsSign[i] = -1;
		}
	}
}

void CalcNewPointPosition(double* pointsLocationPernament,double* pointsLocation, double* velocities,
	int N, int K, double time)
{
#pragma omp parallel for
	for (int i = 0; i < N*K; i++)
	{
		pointsLocation[i] = pointsLocationPernament[i] + velocities[i] * time;
	}
}



double getQuailtyWithOpenMP(int* pointsSign, int* pointsSignSet, int n)
{
	int numOfMisses = 0;
	int i;
#pragma omp parallel for reduction(+:numOfMisses)
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
		w[i] = WTYPE;
}
void updateWValues(double* w, double* pointLocation, int sign, int K, double alpha)
{
	w[K] += (-sign)*alpha;
	for (int i = 0; i < K; i++)
	{
		w[i] += -(sign)* alpha * pointLocation[i];
	}
}

void printResult(double* w, int k, double time, double q)
{

	int i;
	printf("q =   %lf   ", q);
	printf("minimum time =    %lf    \n", time);
	for (i = 0; i < k + 1; i++)
	{
		printf("%lf\n", w[i]);

	}
	
}
void printResultTofile(double* w, int k, double time, double q)
{
	FILE *f = fopen(OUTPUT_FILE, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	fprintf(f, "T minimum: %lf       q =  %lf", time, q);
	for (int i = 0; i < k; i++)
	{
		fprintf(f, "\n%lf", w[i]);
	}
	fclose(f);
}
void copyarray(double * pointsLocation, double * pointsLocationPernament, int size)
{
	for (int i = 0; i < size; i++)
	{
		pointsLocationPernament[i] = pointsLocation[i];
	}
}


