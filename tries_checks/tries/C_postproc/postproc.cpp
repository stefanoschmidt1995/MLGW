#include <iostream>
#include <math.h>       /* sin, cos */

int indices_2D(int i, int j, int D){ //array with (N,D)
	return i*D+j;
}

int indices_3D(int i, int j, int k, int D_1, int D_2){ //array (N, D_1, D_2)
	return i*D_1*D_2+j*D_2+k;
}

extern "C" double* post_process(int N_data, int D_std, int D_us, const double * t_std, const double* t_us, const double* amp, const double* ph, double* m_tot, double* d_L, double* iota, double* phi){
/*
	Given amplitude, phase on a custom grid it computes the h_p, h_c polarizations on another grid, including dependence on luminosity distance, inclination and reference phase.
	t_std is in reduced grid!!! While t_std is in normal grid
*/

	double* sin_phi = new double[N_data];
	double* cos_phi = new double[N_data];

	for(int i=0; i< N_data; i++){
		iota[i] = cos(iota[i]); //iota = cos(iota)
		sin_phi[i] = sin(2*phi[i]);
		cos_phi[i] = cos(2*phi[i]); 
	}

	double m_amp, q_amp;
	double m_ph, q_ph;

		//declaring polarizations
	//default inizialization is to zero: if outside interpolation domain, a value of h = 0 is returned
	double * h = new double[N_data*D_us*2]; //(N, D_us, 2)

	for(int i = 0;i < N_data*D_us*2; i++) h[i]=0;
	double m_tot_std = 20.; 

	//for(int i = 0; i< D_std*N_data; i++) std::cout << amp[i] << " "; //DEBUG 
	//std::cout << std::endl;//DEBUG

	for(int j=0; j<N_data; j++){ //loop on WFs...
		std::cout << j<< " " << m_tot[j] << std::endl;//DEBUG
		int ip = 0;
		int ip_next = 1;
		int i = 0;
		double ph_0 = -1234567.89123456789;
		while(i < D_us){
					//computing interpolation coefficients
			if(t_us[i]/m_tot[j]<t_std[ip_next]){
			m_amp = (amp[indices_2D(j,ip_next,D_std)]-amp[indices_2D(j,ip,D_std)])/(t_std[ip_next]-t_std[ip]);
			q_amp = amp[indices_2D(j,ip,D_std)] - m_amp * t_std[ip];
			m_ph = (ph[indices_2D(j,ip_next,D_std)]-ph[indices_2D(j,ip,D_std)])/(t_std[ip_next]-t_std[ip]);
			q_ph = ph[indices_2D(j,ip,D_std)] - m_ph * t_std[ip];
			}
			while(t_us[i]/m_tot[j]<t_std[ip_next]){
				if(t_us[i]/m_tot[j]>=t_std[ip]){
						//interpolating on reduced grid
					double temp_amp = m_amp*t_us[i]/m_tot[j]+q_amp;
					double temp_ph = m_ph*t_us[i]/m_tot[j]+q_ph;
						//computing phi_0
					if(ph_0 == -1234567.89123456789) ph_0 = temp_ph; //phi_0 is compute only for the first time (UNEFFICIENT!!)
					temp_amp = temp_amp * m_tot[j]/m_tot_std; //applying mass scaling
					temp_ph = temp_ph - ph_0; //scaling back the phase
					//std::cout << temp_ph << std::endl;
					double temp_h_real = temp_amp * cos(temp_ph);
					double temp_h_imag = temp_amp * sin(temp_ph);
					h[indices_3D(j,i,0,D_us,2)] = 1e-21*((1+iota[j]*iota[j])/(2*d_L[j]) * (cos_phi[j]*temp_h_real - sin_phi[j]*temp_h_imag )); //hp
					h[indices_3D(j,i,1,D_us,2)] = 1e-21*((iota[j]/d_L[j]) * (sin_phi[j]*temp_h_real + cos_phi[j]*temp_h_imag )); //hc
					//h[indices_3D(j,i,0,D_us,2)] = 1e-21*temp_amp;h[indices_3D(j,i,1,D_us,2)] = temp_ph;
					//h[indices_3D(j,i,1,D_us,2)] -(j+1); h[indices_3D(j,i,0,D_us,2)] (j+1); //DEBUG
				}
				i +=1;
				if (i >= D_us) {break;}
			}
			ip +=1;
			ip_next +=1;
			if(ip_next == D_std) break; //exiting the loop when t_std is over
		}
	}
	delete sin_phi;
	delete cos_phi;
	return h;
} //*/














