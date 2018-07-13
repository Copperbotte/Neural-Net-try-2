// Neural Net try 2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include <Windows.h>
using namespace std;

unsigned long xorshfnums[3] = {123456789, 362436069, 521288629}; // I think these are random, and can be randomized using a seed
unsigned long xorshf96(void) // YOINK http://stackoverflow.com/questions/1640258/need-a-fast-random-generator-for-c
{          //period 2^96-1
	unsigned long t;
	xorshfnums[0] ^= xorshfnums[0] << 16;
	xorshfnums[0] ^= xorshfnums[0] >> 5;
	xorshfnums[0] ^= xorshfnums[0] << 1;

	t = xorshfnums[0];
	xorshfnums[0] = xorshfnums[1];
	xorshfnums[1] = xorshfnums[2];
	xorshfnums[2] = t ^ xorshfnums[0] ^ xorshfnums[1];

	return xorshfnums[2];
}
double xorshfdbl(void)
{ // Double: sign bit, 11 exponent bits, 52 fraction bits,  0x3ff0000000000000 = Exponent and Power section, equivelant to 1
	unsigned long long x = 0x3ff0000000000000 | ((unsigned long long)xorshf96() << 20); //xorshft92 is 32 bits long, 32 - 52 = 20 bits shifted
	return *(double*)&x - 1.0;
}
unsigned long *xorshfdata(void)
{
	return xorshfnums;
}


double Tanh(double In)
{
	// Sigmoid: tanh
	//double P = exp(In);
	//double N = exp(-In);
	//In = (P-N)/(P+N);
	// This optomization prioritizes zeros over infinites
	In *= 2;
	if(0 < In) // N/P is small
	{
		In = exp(-In);
		In = (1-In)/(1+In);
	}
	else // T < 0 P/N is small
	{
		In = exp(In);
		In = (In-1)/(In+1);
	}
	return In;
}

double dTanh(double Sig)
{
	// Sigmoid: tanh
	// Sig' = 1-Sig^2
	return 1.0-Sig*Sig;
}

double Err(int Layers, int* Neuron_Count, double** Neuron, double* Expected)
{
	//cout << "Error from Data\n"; 
	double error = 0.0;
	for(int i=0;i<Neuron_Count[Layers-1];++i) // Matrix multiply
	{
		double Delta = Neuron[Layers-1][i]-Expected[i];
		//cout << Delta*Delta << ", ";
		error += Delta*Delta;
	}
	//cout << "\nError: " << error << '\n';
	return error;
}

int _tmain(int argc, _TCHAR* argv[])
{
	xorshfnums[0] = GetTickCount();

	int Neuron_Count[] = {2,4,1};
	int Layers = sizeof(Neuron_Count)/sizeof(int);

	double (*Sigmoid)(double) = Tanh;
	double (*dSigmoid)(double) = dTanh;

	//int Neuron_Count[] = {2,4,1};
	//   *   *
	//  * * * *
	//     *

	// Construct initialization

	double** Weights = new double*[Layers-1]; // Layers can never be zero unless the user forgot their neurons.
	double** Numer = new double*[Layers-1];
	double** Denom = new double*[Layers-1];
	double** Neuron = new double*[Layers];
	double** dNum = new double*[Layers]; // Derivative, used later on
	double** dDen = new double*[Layers];
	for(int i=0;i<Layers-1;++i)
	{
		unsigned int n = (Neuron_Count[i]+1)*Neuron_Count[i+1]; // Bias node
		Weights[i] = new double[n];
		Numer[i] = new double[n];
		Denom[i] = new double[n];
		for(int dst=0;dst<Neuron_Count[i+1];++dst)
			for(int src=0;src<Neuron_Count[i]+1;++src)							 // xorshf96 is a random number generator.
				Weights[i][(Neuron_Count[i]+1)*dst+src] = (xorshfdbl()*2.0)-1.0; // the dbl variant maps the number to 1.0-0.0 with double precision.
	}
	for(int i=0;i<Layers;++i)
	{
		int n = Neuron_Count[i];
		if(i<Layers-1) ++n; // Bias node
		Neuron[i] = new double[n];
		dNum[i] = new double[n];
		dDen[i] = new double[n];
	}

	// Data Initialization
	int Data_Count = 100; // 1 data point :^)
	double **Initial = new double*[Data_Count];
	double **Expected = new double*[Data_Count];
	for(int i=0;i<Data_Count;++i)
	{
		Initial[i] = new double[Neuron_Count[0]];
		Expected[i] = new double[Neuron_Count[Layers-1]]; // Expected value is always the same as the final neuron
		double theta = 2.0*3.1415928565897932384*(double)i/10.0; 
		Initial[i][0] = cos(theta)*((double)(i/10));
		Initial[i][1] = sin(theta)*((double)(i/10));
		Expected[i][0] = sqrt(Initial[i][0]*Initial[i][0]+Initial[i][1]*Initial[i][1]);
	}

	//double Init[] = {1.0,2.0};
	//double fin[] = {0.5};
	//memcpy(Initial[0],Init,2*sizeof(double));
	//memcpy(Expected[0],fin,1*sizeof(double));

	// Initialize backprop cycle
	// Clear backprop accumulators
	for(int z=0;z<20;++z)
	{
		for(int layer=Layers-2;0<=layer;--layer) // Weights
			for(int src=0;src<Neuron_Count[layer]+1;++src)
				for(int dst=0;dst<Neuron_Count[layer+1];++dst)
				{
					int pos = (Neuron_Count[layer]+1)*dst+src;
					Numer[layer][pos] = 0.0;
					Denom[layer][pos] = 0.0;
				}

		// Setup bias
		// This value should never change, but should be included in case of errors.
		for(int layer=0;layer<Layers-1;++layer)
			Neuron[layer][Neuron_Count[layer]] = 1.0;

		double error = 0.0;

		// Forward Propogate
		// This propogates a neuron's signal forward in the network by one stage.
		// The bias node should be the final element in the vector, to allow for memory copying and forward propogation without arithmatic.
		// the bias node is always 1, and should never have backward attached weights.

		//cout << fixed;
		//for(int i=0;i<Neuron_Count[0]+1;++i)
		//	cout << setprecision(5) << Neuron[0][i] << ", ";
		//cout << '\n';
		for(int m=0;m<Data_Count;++m)
		{
			memcpy(Neuron[0],Initial[m],Neuron_Count[0]*sizeof(double));
			for(int layer=0;layer<Layers-1;++layer)
			{
				for(int dst=0;dst<Neuron_Count[layer+1];++dst)
				{
					double *T = Neuron[layer+1] + dst; // Target node
					*T = 0.0; // Clear accumulator
					for(int src=0;src<Neuron_Count[layer]+1;++src) // Matrix multiply, source has bias included.
					{
						*T += Weights[layer][(Neuron_Count[layer]+1)*dst+src]*Neuron[layer][src];
			//			cout << setprecision(5) << Weights[layer][(Neuron_Count[layer]+1)*dst+src] << ", ";
					}
			//		cout << " | " << setprecision(5) << *T;
					*T = Sigmoid(*T);
			//		cout << " | " << setprecision(5) << *T << '\n';
				}
			//	cout << '\n';
			}

			error += Err(Layers, Neuron_Count, Neuron, Expected[m]);

			//for(int i=0;i<Neuron_Count[Layers-1];++i)
			//	cout << setprecision(5) << Expected[0][i] << ", ";
			//cout << '\n';

			// Backward Propogate
			// This propogate should be one in two steps: propogate weight, then neuron.
			// These steps may be possible to merge into one step.
			// dN2/dW = N1, so the weight's derivative is its associated backward neuron.
			// Weight order doesn't matter.
			// Includes bias weights.
			// dN2/dN1 = W, summed over all neurons to that N1.
			// Operation is identical to a transposed matrix multiplication.
			// Excludes bias neuron, first, and last stage.

			// Derivative setup
			for(int i=0;i<Neuron_Count[Layers-1];++i)
			{
				double dSig = dSigmoid(Neuron[Layers-1][i]);
				dNum[Layers-1][i] = dSig*(Neuron[Layers-1][i] - Expected[m][i]); // Discrete derivative for the numerator
				dDen[Layers-1][i] = dSig; // Multiplicitive null derivative for the denominator
			}

			for(int layer=Layers-2;0<layer;--layer) // BACKpropogation, doesn't effect the first layer
				for(int src=0;src<Neuron_Count[layer];++src) // Loop thorough source neurons first, they're needed for the weights.
				{	//  D = W*S, W is Transposed
					dNum[layer][src] = 0.0; // clear derivative accumulators
					dDen[layer][src] = 0.0;
					for(int dst=0;dst<Neuron_Count[layer+1];++dst) // Transposed matrix multiply
					{	// dW/dN2 = N1, dN1/dN2 = W
						int pos = Neuron_Count[layer+1]*src+dst;
						double dSig = dSigmoid(Neuron[layer+1][dst])*Neuron[layer+1][dst];
						double der = Weights[layer][pos]*Neuron[layer+1][dst]*dSig;
						dNum[layer][src] += dNum[layer+1][dst]*der;
						dDen[layer][src] += dDen[layer+1][dst]*der;
					}
				}

			for(int layer=Layers-2;0<=layer;--layer) // Weights
				for(int dst=0;dst<Neuron_Count[layer+1];++dst)
					for(int src=0;src<Neuron_Count[layer]+1;++src)
					{	// dW/dN2 = N1, dN1/dN2 = W
						int pos = (Neuron_Count[layer]+1)*dst+src;
						double bubble = Neuron[layer+1][dst];
						double dSig = dSigmoid(Neuron[layer+1][dst])*Neuron[layer+1][dst];
						double der = Neuron[layer+1][dst]*dSig;
						Numer[layer][pos] += dNum[layer+1][dst]*der;
						der *= dDen[layer+1][dst];
						Denom[layer][pos] += der*der; // Mean square of the derivatives
					}
		}

		//system("pause");

		cout << fixed;

		// Forward Propogate derivative to update
		for(int layer=0;layer<Layers-1;++layer)
		{
			for(int dst=0;dst<Neuron_Count[layer+1];++dst)
			{
				for(int src=0;src<Neuron_Count[layer]+1;++src)
				{
					int pos = (Neuron_Count[layer]+1)*dst+src;
					if(Denom[layer][pos] < 1.0)
						Weights[layer][pos] -= Numer[layer][pos];//Denom[layer][pos];
					else
						Weights[layer][pos] -= Numer[layer][pos]/Denom[layer][pos];
					cout << fixed;
					cout << setprecision(5) << Weights[layer][pos] << ", ";
				}
				cout << '\n';
			}
			cout << '\n';
		}
		
		cout << "Step " << z << '\n';
		cout << fixed;
		//double e = Err(Layers,Neuron_Count,Neuron,Expected);
		error /= Data_Count;
		cout << "Error: " << error << '\n';
		//if(z%100 == 0)
			system("pause");

		if(error < 0.000001) break;
		if(error != error) break;

		cout << '\n';
	}
	// Backward Propogate
	// N2 = Sigmoid{ W*N1 }
	// dW/dN1 = N1
	// dN2/dN1 = sum(Sig'{W*N1}*W for all W)
	//
	// Euler's Method
	// -y0 = (x1-x0)y'
	// -y0/y' = x1-x0
	// x1 = x0 - y0/y'
	// x -= y/y'
	// Euler's Method (2nd derivative)
	// x -= y'/y"
	//
	// Neural Error Sum
	// y = sum((Generated - Expected)^2, for all neurons, for all test cases)
	// y'= sum(2(G-E)G', for all test cases) // Derivative is on a per neuron basis.
	// y"= sum(2(G-E)G" + 2G'^2, for all test cases)
	// x -= sum(G'(G-E))/sum(G'^2 + G"(G-E))
	// G" will eventually source from a zero, as G's input data is a set of linear functions.
	// x -= sum(G'(G-E))/sum(G'^2)
	// 
	// G' IS backpropogation
	// For example, unbiased, one node weight
	// G' = N
	// x -= sum(N(WN-E))/sum(N^2)
	// 
	// biased, one node weight
	// x -= sum(N*S(WN)(WN-E))/sum((N*S(WN))^2)
	// N' = W
	// This can be found via a transpose.


	// garbage day
	for(int i=0;i<Layers-1;++i)
	{
		delete[] Weights[i];
		delete[] Numer[i];
		delete[] Denom[i];
	}
	delete[] Weights;
	delete[] Numer;
	delete[] Denom;
	Weights = Numer = Denom = nullptr;

	for(int i=0;i<Layers;++i)
	{
		delete[] Neuron[i];
		delete[] dNum[i];
		delete[] dDen[i];
	}
	delete[] Neuron;
	delete[] dNum;
	delete[] dDen;
	Neuron = dNum = dDen = nullptr;

	for(int i=0;i<Data_Count;++i)
	{
		delete[] Initial[i];
		delete[] Expected[i];
	}
	delete[] Initial;
	delete[] Expected;
	Initial = Expected = nullptr;

	system("pause");
	return 0;
}

