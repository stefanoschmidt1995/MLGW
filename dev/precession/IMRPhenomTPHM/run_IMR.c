#include <lal/LALSimIMR.h>
#include <lal/LALSimInspiral.h>
#include <lal/SphericalHarmonics.h>
#include <lal/LALConstants.h>

// Standard C 
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/*
OSS: The NP modes are the same in the L0 frame and in the J0 frame. This means that we can generate a NP dataset that works for both cases. The frame of the P model will be determined by the form of the Euler angles

*/

/*
Call structure for ChooseTDModes with IMRPhenomTPHM
XLALSimInspiralChooseTDModes
	|
	|
	V
XLALSimIMRPhenomTPHM_ChooseTDModes
	|
	|
	V
XLALSimIMRPhenomTPHM_L0Modes
	|
	|
	V
XLALSimIMRPhenomTPHM_JModes + some stuff
	|
	|
	V
XLALSimIMRPhenomTPHM_CoprecModes
(It does all the job of creating the TD modes and the Euler angles)

##############

*/


int main(int argc, char *argv[]){
	REAL8TimeSeries *hplus;
	REAL8TimeSeries *hcross;
	SphHarmTimeSeries *hlmJ;
	SphHarmTimeSeries *hlmJ_NP;
	REAL8TimeSeries *alphaTS;
	REAL8TimeSeries *cosbetaTS;
	REAL8TimeSeries *gammaTS;
	REAL8 af = -100.; 
	REAL8 iota = 0.;
    REAL8 distance = 1e6*LAL_PC_SI;

	REAL8 m1 = 50.*LAL_MSUN_SI;
	REAL8 m2 = 30.*LAL_MSUN_SI;
	REAL8 S1x = -0.3;
	REAL8 S1y = .2;
	REAL8 S1z = 0.4;
	REAL8 S2x = -0.5;
	REAL8 S2y = -0.1;
	REAL8 S2z = 0.4;
	
	REAL8 deltaT = 1e-3;
	REAL8 f_min = 10.;
    	
	if (argc == 11){ //parameters can be given from command line
		m1 = strtod(argv[1],NULL)*LAL_MSUN_SI;
		m2 = strtod(argv[2],NULL)*LAL_MSUN_SI; //pay attention here!!! It is m2, not m1!!!
		S1x = strtod(argv[3],NULL);
		S1y = strtod(argv[4],NULL);
		S1z = strtod(argv[5],NULL);
		S2x = strtod(argv[6],NULL);
		S2y = strtod(argv[7],NULL);
		S2z = strtod(argv[8],NULL);

		f_min = strtod(argv[9],NULL);
		deltaT = strtod(argv[10],NULL);
		
    }

	//printf("%f %f %f %f\n", m1/LAL_MSUN_SI, m2/LAL_MSUN_SI, S1z, S2z);
    
    REAL8 f_ref = f_min;
    
    	//the modes shall be divided by the factor M/R
	REAL8 scale_factor = (m1+m2)/distance * 7.42591549e-28;
	scale_factor *= (m1*m2/((m1+m2)*(m1+m2)));
    
	LALDict *LALparams = XLALCreateDict();
	INT4 precVer = XLALSimInspiralWaveformParamsLookupPhenomXPrecVersion(LALparams);
	//printf("PrecVers %i \n", precVer); //DEBUG

		//getting NP modes (in either L0 or J0 frame)
	XLALSimIMRPhenomTPHM_CoprecModes(
		&hlmJ_NP,            	//< [out] Modes in the intertial J0=z frame
		&alphaTS,  			//< [out] Precessing Euler angle alpha 
		&cosbetaTS,   		//< [out] Precessing Euler angle beta 
		&gammaTS,  			//< [out] Precessing Euler angle gamma 
		&af,					//< [out] Final spin 
		m1,          //< Mass of companion 1 (kg) 
		m2,			//< Mass of companion 2 (kg) 
		S1x,				    //< x-component of the dimensionless spin of object 1 
		S1y,				    //< y-component of the dimensionless spin of object 1 
		S1z,				    //< z-component of the dimensionless spin of object 1 
		S2x,				    //< x-component of the dimensionless spin of object 2 
		S2y,				    //< y-component of the dimensionless spin of object 2 
		S2z,				    //< z-component of the dimensionless spin of object 2 
		distance,			 //< distance of source (m) 
		iota,				     //< inclination of source (rad) 
		deltaT,				 //< sampling interval (s) 
		f_min,				  //< starting GW frequency (Hz) 
		f_ref,				  //< reference GW frequency (Hz)
		0.,				     //< reference orbital phase (rad)  
		LALparams,       //< LAL dictionary containing accessory parameters 
		0              //< Flag for calling only IMRPhenomTP (dominant 22 coprec mode only) 
		);

		//getting precessing modes (in J0 frame)
	XLALSimIMRPhenomTPHM_JModes(
		&hlmJ,            	//< [out] Modes in the intertial J0=z frame
		&alphaTS,  			//< [out] Precessing Euler angle alpha 
		&cosbetaTS,   		//< [out] Precessing Euler angle beta 
		&gammaTS,  			//< [out] Precessing Euler angle gamma 
		&af,					//< [out] Final spin 
		m1,          //< Mass of companion 1 (kg) 
		m2,			//< Mass of companion 2 (kg) 
		S1x,				    //< x-component of the dimensionless spin of object 1 
		S1y,				    //< y-component of the dimensionless spin of object 1 
		S1z,				    //< z-component of the dimensionless spin of object 1 
		S2x,				    //< x-component of the dimensionless spin of object 2 
		S2y,				    //< y-component of the dimensionless spin of object 2 
		S2z,				    //< z-component of the dimensionless spin of object 2 
		distance,			 //< distance of source (m) 
		iota,				     //< inclination of source (rad) 
		deltaT,				 //< sampling interval (s) 
		f_min,				  //< starting GW frequency (Hz) 
		f_ref,				  //< reference GW frequency (Hz)
		0.,				     //< reference orbital phase (rad)  
		LALparams,       //< LAL dictionary containing accessory parameters 
		0              //< Flag for calling only IMRPhenomTP (dominant 22 coprec mode only) 
		); //*/
	
	if (0){ //looking into XLALSimIMRPhenomTPHM_L0Modes and XLALSimIMRPhenomTPHM_ChooseTDModes
		/*printf("Using XLALSimIMRPhenomTPHM_L0Modes\n");
		XLALSimIMRPhenomTPHM_L0Modes(
		&hlmJ,            	//< [out] Modes in the intertial J0=z frame
			m1,          //< Mass of companion 1 (kg) 
			m2,			//< Mass of companion 2 (kg) 
			S1x,				    //< x-component of the dimensionless spin of object 1 
			S1y,				    //< y-component of the dimensionless spin of object 1 
			S1z,				    //< z-component of the dimensionless spin of object 1 
			S2x,				    //< x-component of the dimensionless spin of object 2 
			S2y,				    //< y-component of the dimensionless spin of object 2 
			S2z,				    //< z-component of the dimensionless spin of object 2 
			distance,			 //< distance of source (m) 
			iota,				     //< inclination of source (rad) 
			deltaT,				 //< sampling interval (s) 
			f_min,				  //< starting GW frequency (Hz) 
			f_ref,				  //< reference GW frequency (Hz)
			0.,				     //< reference orbital phase (rad)  
			LALparams,       //< LAL dictionary containing accessory parameters 
			0              //< Flag for calling only IMRPhenomTP (dominant 22 coprec mode only) 
		);//*/
		
		printf("Using XLALSimIMRPhenomTPHM_ChooseTDModes\n");
		hlmJ = XLALSimIMRPhenomTPHM_ChooseTDModes(
			m1,          //< Mass of companion 1 (kg) 
			m2,			//< Mass of companion 2 (kg) 
			S1x,				    //< x-component of the dimensionless spin of object 1 
			S1y,				    //< y-component of the dimensionless spin of object 1 
			S1z,				    //< z-component of the dimensionless spin of object 1 
			S2x,				    //< x-component of the dimensionless spin of object 2 
			S2y,				    //< y-component of the dimensionless spin of object 2 
			S2z,				    //< z-component of the dimensionless spin of object 2 
			distance,			 //< distance of source (m) 
			deltaT,				 //< sampling interval (s) 
			f_min,				  //< starting GW frequency (Hz) 
			f_ref,				  //< reference GW frequency (Hz)
			LALparams       //< LAL dictionary containing accessory parameters 
		);
	}

	/*Structure of SphHarmTimeSeries
		- l
		- m
		- next (SphHarmTimeSeries)
		- mode (Complex TimeSeries):
			- data: (Complex sequence)
				- data
				- lenght
			- deltaT
			-f_0
	*/

	//Loop over the modes and saving stuff to file
	//Header of the file is (should be transposed in python):
	// alpha | beta | gamma | lm.real | lm.imag
	FILE* f_P = fopen("modes_P.txt", "w");
	FILE* f_NP = fopen("modes_NP.txt", "w");
	FILE* f_angles = fopen("angles.txt", "w");
	FILE* fh = fopen("header.txt", "w");
	int count = 0; //to count the index in the header
	
	//printf("%e %e %e\n", alphaTS->deltaT,hlmJ->mode->deltaT, hlmJ_NP->mode->deltaT); //DEBUG
	
	fprintf(fh, "angles.txt \t |0: times |1: alpha | 2: cos(beta) | 3: gamma");
	for (int j=0; j<alphaTS->data->length; j++) //saving time
		fprintf(f_angles, "%e ", alphaTS->deltaT*j);
	fprintf(f_angles, "\n");
	for (int j=0; j<alphaTS->data->length; j++) //saving alpha
		fprintf(f_angles, "%e ", alphaTS->data->data[j]);
	fprintf(f_angles, "\n");
	for (int j=0; j<cosbetaTS->data->length; j++) //saving cosbeta
		fprintf(f_angles, "%e ", cosbetaTS->data->data[j]);
	fprintf(f_angles, "\n");
	for (int j=0; j<gammaTS->data->length; j++) //saving gamma
		fprintf(f_angles, "%e ", gammaTS->data->data[j]);
	fprintf(f_angles, "\n");
	
	fprintf(fh, "\nmodes_P.txt \t |%i: times", count);
	
	for (int j=0; j<hlmJ->mode->data->length; j++)
		fprintf(f_P, "(%e+%ej) ",  hlmJ->mode->deltaT*j, 0.);
	fprintf(f_P,"\n");	
	
	for(SphHarmTimeSeries* i = hlmJ; i->next != NULL; i=i->next){
		//printf("P: %i,%i \n", i->l, i->m); //DEBUG
		count++;
		//printf("len data %i \n", i->mode->data->length);
		fprintf(fh, " |%i: (%i,%i) ", count, i->l, i->m);
		
		//COMPLEX16 sph = XLALSpinWeightedSphericalHarmonic(iota, 0., -2, i->l, i->m);

		for (int j=0; j<i->mode->data->length; j++)
			fprintf(f_P, "(%e+%ej) ", creal(i->mode->data->data[j])/scale_factor,  cimag(i->mode->data->data[j])/scale_factor);
		fprintf(f_P,"\n");
	}

	count = 0;
	fprintf(fh, "\nmodes_NP.txt \t |%i: times", count);
	
	for (int j=0; j<hlmJ_NP->mode->data->length; j++)
		fprintf(f_NP, "(%e+%ej) ",  hlmJ_NP->mode->deltaT*j, 0.);
	fprintf(f_NP,"\n");	
	
	for(SphHarmTimeSeries* i = hlmJ_NP; i->next != NULL; i=i->next){
		//printf("NP: %i,%i \n", i->l, i->m); //DEBUG
		count++;
		//printf("len data %i \n", i->mode->data->length);
		fprintf(fh, " |%i: (%i,%i) ", count, i->l, i->m);

		for (int j=0; j<i->mode->data->length; j++)
			fprintf(f_NP, "(%e+%ej) ", creal(i->mode->data->data[j])/scale_factor,  cimag(i->mode->data->data[j])/scale_factor);
		fprintf(f_NP,"\n");
	}

	fprintf(fh, "\n");
	fclose(f_P);
	fclose(f_angles);
	fclose(fh);

	if (0){ //modes in the L0 frame
	SphHarmTimeSeries* hlmJ_TDModes = XLALSimIMRPhenomTPHM_ChooseTDModes(
		m1,          //< Mass of companion 1 (kg) 
		m2,			//< Mass of companion 2 (kg) 
		S1x,				    //< x-component of the dimensionless spin of object 1 
		S1y,				    //< y-component of the dimensionless spin of object 1 
		S1z,				    //< z-component of the dimensionless spin of object 1 
		S2x,				    //< x-component of the dimensionless spin of object 2 
		S2y,				    //< y-component of the dimensionless spin of object 2 
		S2z,				    //< z-component of the dimensionless spin of object 2 
		distance,			 //< distance of source (m) 
		deltaT,				 //< sampling interval (s) 
		f_min,				  //< starting GW frequency (Hz) 
		f_ref,				  //< reference GW frequency (Hz)
		LALparams       //< LAL dictionary containing accessory parameters 
		);
	
	}

	if(0){	// This calls the h_p/h_c waveform
		int status = XLALSimIMRPhenomTPHM(
		&hplus,			    //< +-polarization waveform 
		&hcross,			   //< x-polarization waveform 
		m1,			       //< mass of companion 1 (kg) 
		m2,			       //< mass of companion 2 (kg) 
		S1x,			      //< x-component of the dimensionless spin of object 1 
		S1y,			      //< y-component of the dimensionless spin of object 1 
		S1z,			      //< z-component of the dimensionless spin of object 1 
		S2x,			      //< x-component of the dimensionless spin of object 2 
		S2y,			      //< y-component of the dimensionless spin of object 2 
		S2z,			      //< z-component of the dimensionless spin of object 2 
		distance,			 //< distance of source (m) 
		iota,			       //< inclination of source (rad) 
		deltaT,			   //< sampling interval (s) 
		f_min,			    //< starting GW frequency (Hz) 
		f_ref,			    //< reference GW frequency (Hz)
		0.,			       //< reference orbital phase (rad)  
		LALparams			 //< LAL dictionary containing accessory parameters 
		);
	}

	if(0){	// This calls the h_p/h_c waveform
	    Approximant approximant = XLALSimInspiralGetApproximantFromString("SEOBNRv4P");

		LALDict *LALparams = XLALCreateDict();

		int status = XLALSimInspiralChooseTDWaveform(
			&hplus,                    /**< +-polarization waveform */
			&hcross,                   /**< x-polarization waveform */
			m1,                       /**< mass of companion 1 (kg) */
			m2,                       /**< mass of companion 2 (kg) */
			S1x,                      /**< x-component of the dimensionless spin of object 1 */
			S1y,                      /**< y-component of the dimensionless spin of object 1 */
			S1z,                      /**< z-component of the dimensionless spin of object 1 */
			S2x,                      /**< x-component of the dimensionless spin of object 2 */
			S2y,                      /**< y-component of the dimensionless spin of object 2 */
			S2z,                      /**< z-component of the dimensionless spin of object 2 */
			distance,                 /**< distance of source (m) */
			iota,                       /**< inclination of source (rad) */
			0.,                       /**< reference orbital phase (rad) */
			0.,                       /**< longitude of ascending nodes, degenerate with the polarization angle, Omega in documentation */
			0.,                       /**< eccentricity at reference epoch */
			0.,                       /**< mean anomaly of periastron */
			deltaT,                         /**< sampling interval (s) */
			f_min,                          /**< starting GW frequency (Hz) */
			f_ref,                                /**< reference GW frequency (Hz) */
			LALparams,                         /**< LAL dictionary containing accessory parameters */
			approximant               /**< post-Newtonian approximant to use for waveform production */
		); 
		for (int j=0; j< hplus->data->length; j++)
			printf("%f ,", hplus->data->data[j]);
	}

	//printf("IMR runned: take a look at header.txt, angles.txt, modes_P.txt, modes_NP.txt\n");
	return 0;

}



