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

int get_angles(
  REAL8TimeSeries *alphaTS,  /**< [out] Precessing Euler angle alpha */
  REAL8TimeSeries *betaTS,   /**< [out] Precessing Euler angle beta */
  REAL8TimeSeries *gammaTS,  /**< [out] Precessing Euler angle gamma */
  REAL8 m1,                /**< Mass of companion 1 (kg) */
  REAL8 m2,                /**< Mass of companion 2 (kg) */
  REAL8 S1x,                /**< x component of primary spin*/
  REAL8 S1y,                /**< y component of primary spin*/
  REAL8 S1z,                /**< z component of primary spin */
  REAL8 S2x,                /**< x component of secondary spin*/
  REAL8 S2y,                /**< y component of secondary spin*/
  REAL8 S2z,                /**< z component of secondary spin */
  REAL8 deltaT,               /**< sampling interval (s) */
  REAL8 fmin,               /**< starting GW frequency (Hz) */
  REAL8 fRef               /**< reference GW frequency (Hz) */
  ){
  
  	SphHarmTimeSeries *hlmJ;
  	REAL8 af;
	LALDict *LALparams = XLALCreateDict();
  	XLALSimIMRPhenomTPHM_CoprecModes(
		&hlmJ,            	//< [out] Modes in the intertial J0=z frame
		&alphaTS,  			//< [out] Precessing Euler angle alpha 
		&betaTS,   		//< [out] Precessing Euler angle beta 
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
		1e6*LAL_PC_SI,			 //< distance of source (m) 
		0.,				     //< inclination of source (rad) 
		deltaT,				 //< sampling interval (s) 
		fmin,				  //< starting GW frequency (Hz) 
		fRef,				  //< reference GW frequency (Hz)
		0.,				     //< reference orbital phase (rad)  
		LALparams,       //< LAL dictionary containing accessory parameters 
		0              //< Flag for calling only IMRPhenomTP (dominant 22 coprec mode only) 
		);

		//cosbeta should be beta nut I cannot link it!!
	//for (int j=0; j<(betaTS)->data->length; j++) //saving cosbeta
	//	betaTS->data->data[j] = acos(betaTS->data->data[j]);

	//LALFree(&hlmJ);
	return 0;
	}

int main(int argc, char *argv[]){
	SphHarmTimeSeries *hlmJ;
	REAL8TimeSeries *alphaTS;
	REAL8TimeSeries *betaTS;
	REAL8TimeSeries *gammaTS;

		//default values for spin
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
  
   	REAL8 f_ref = f_min;
    
    get_angles(alphaTS, betaTS, gammaTS,
    	m1, m2, S1x, S1y, S1z, S2x, S2y,S2z,
    	deltaT, f_min, f_ref
    	);

	
	
	
}
