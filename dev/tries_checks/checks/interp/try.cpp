#include <iostream>
#include <cmath>
#include <chrono>  // for high_resolution_clock

double* interp(int N_x, int N_xp, double* x, double* xp, double* yp){
	int ip = 0;
	int ip_next = 1;
	int i = 0;
	double * y = new double[N_x]; //you should all initialize them to zero!!

	while(i < N_x){
		double m = (yp[ip_next]-yp[ip])/(xp[ip_next]-xp[ip]);
		double q = yp[ip] - m * xp[ip];
		while(x[i]<xp[ip_next]){
			if(x[i]>=xp[ip])
				y[i] = m*x[i]+q;
			i +=1;
			if (i >= N_x) {break;}
			//std::cout << i << " " <<ip <<std::endl;
		}
		ip +=1;
		ip_next +=1;
		if(ip_next == N_xp)
			break;
	}

	return y;
}

int main(){
	long long int N_x = 500000;
	int N_xp = 100;
	
	double xp[N_xp];
	double* x = new double[N_x];
	double yp[N_xp];

	std::cout <<"Number of points: "<< N_x <<std::endl;

	for(int i=0; i< N_x; i++){
		x[i] = (double)i/((double)N_x)*100.;
	}

	for(int i=0; i< N_xp; i++){
		xp[i] = (double)i;
		yp[i] = exp(xp[i]/100.);
	}
	
	auto start = std::chrono::high_resolution_clock::now();

	double* y = interp(N_x,N_xp, x, xp, yp);

	auto finish = std::chrono::high_resolution_clock::now();

	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
	std::cout << "elapsed time: "<< elapsed<<std::endl;

	return 0;
}










