#include <iostream>
//	import_array()  //what is that?????

//double* interp(int N_x, int N_xp, double* x, double* xp, double* yp){
extern "C" double* interp(int N_x, int N_xp, const double * x, const double* xp, const double* yp, double left, double right){

	double * y = new double[N_x];
	for(int i = 0; i<N_x; i++) y[i] =left;

	int ip = 0;
	int ip_next = 1;
	int i = 0;

	while(i < N_x){
		double m = (yp[ip_next]-yp[ip])/(xp[ip_next]-xp[ip]);
		double q = yp[ip] - m * xp[ip];
		//std::cout << m << " "<< q<<std::endl;
		while(x[i]<xp[ip_next]){
			if(x[i]>=xp[ip])
				y[i] = m*x[i]+q;
			//std::cout << y[i] << std::endl; // debug: OK
			i +=1;
			if (i >= N_x) {break;}
			//std::cout << i << " " <<ip <<std::endl;
		}
		ip +=1;
		ip_next +=1;
		if(ip_next == N_xp){
			while(i<N_x){
				y[i] = right;
				i++;
			}
			break;
		}
	}

	return y;
}

int indices(int i, int j, int row_len){
	return i*row_len+j;
}

extern "C" double* interp_N(int N_x, int N_xp, int N_data, const double * x, const double* xp, const double* yp, const double* left, const double* right){

	double* y = new double[N_data*N_x];
	for(int j =0;j<N_data; j++)
		for(int i = 0; i<N_x; i++)
			y[indices(j,i,N_x)] =left[j];

	int ip = 0;
	int ip_next = 1;
	int i = 0;

	double* m = new double[N_data];
	double* q = new double[N_data];

	while(i < N_x){
		for(int j=0; j<N_data; j++){
			m[j] = (yp[indices(j,ip_next,N_xp)]-yp[indices(j,ip,N_xp)])/(xp[ip_next]-xp[ip]);
			q[j] = yp[indices(j,ip,N_xp)] - m[j] * xp[ip];
		}
		while(x[i]<xp[ip_next]){
			if(x[i]>=xp[ip])
				for(int j=0; j<N_data; j++) y[indices(j,i,N_x)] = m[j]*x[i]+q[j];
			i +=1;
			if (i >= N_x) {break;}
		}
		ip +=1;
		ip_next +=1;
		if(ip_next == N_xp){ //exiting the loop and setting value at the right
			while(i<N_x){
				for(int j=0; j<N_data;j++)
					y[indices(j,i,N_x)] = right[j];
				i++;
			}
			break;
		}
	}

	delete m;
	delete q;
	return y;
} //*/














