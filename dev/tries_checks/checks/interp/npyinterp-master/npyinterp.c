/***
 * Fast interpolation for numpy arrays (see README)
 *
 * Author: Johannes Buchner (C) 2013-2015
 */

#include<stdio.h>
#include<Python.h>
#include<numpy/arrayobject.h>
#include<assert.h>

npy_intp binary_search(const double key, const double arr [], const npy_intp len)
{
    npy_intp imin = 0;
    npy_intp imax = len;

    if (key > arr[len - 1]) {
        return len;
    }
    while (imin < imax) {
        npy_intp imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) {
            imin = imid + 1;
        }
        else {
            imax = imid;
        }
    }
    return imin - 1;
}

#define IFVERBOSE if(0)
#define adouble double
#define bdouble double

/**
 * Main interpolation+integration function.
 * 
 * Uses the data arrays xp, yp to interpolate values (yp) defined at certain positions (xp).
 * It computes the integral from xa to xb using linear, piecewise interpolation.
 * This can be done for many values of xa, xb, which are defined as bins:
 *   leftp, rightp define the borders of the bins (of length m)
 *   zp  is where the results of the interpolation+integration should be stored.
 * 
 * Parameters regarding where to interpolate:
 * leftp:  double array: coordinate of lower bin border
 * rightp: double array: coordinate of upper bin border
 * zp:     double array: storage for the result. 
 * m:      int: size of each of the three arrays
 * Parameters regarding the data used for interpolation:
 * xp:     double array: coordinate
 * yp:     double array: value
 * n:      double array: size of the lookup table
 * 
 */
int interpolate_integrate(const void * leftp, const void * rightp, void * zp, int m,
	const void * xp, const void * yp, int n) {
	int i;
	int j;
	int k;
	const adouble * left = (adouble*) leftp;
	const adouble * right = (adouble*) rightp;
	const bdouble * x = (bdouble*) xp;
	const bdouble * y = (bdouble*) yp;
	adouble * z = (adouble*) zp;
	double yleft, yright;
	double wleft, wright;
	/* find start */
	if (left[0] < x[0]) {
		fprintf(stderr, "left (%f) is below of x-range(%f...)\n", left[0], x[0]);
		return 1;
	}
	j = binary_search(left[0], x, n);
	IFVERBOSE printf("binary search found %i <%f|%f|%f>\n", j, x[j], left[0], x[j+1]);
	k = j + 1;
	if (j + 1 >= m) {
		fprintf(stderr, "left (%f) is above of x-range(...%f)\n", left[0], x[m-1]);
		return 1;
	}
	
	/* assumptions: 
	 * left > right 
	 * x,left,right are increasing
	 */
	 
	/* for each left/right, 
	 * set z to the corresponding y-value */
	for (i = 0; i < m; i++) {
		IFVERBOSE printf("looking for %f..%f (%d of %d)\n", left[i],right[i], i, m);
		for(;left[i]  > x[j+1] && j < n; j++) {
			IFVERBOSE printf("  moved left  to %d: %f\n", j, x[j]);
		}
		for(;right[i] > x[k+1] && k < n; k++) {
			IFVERBOSE printf("  moved right to %d: %f\n", k, x[k]);
		}
		if (k == n || j == n) {
			fprintf(stderr, "box (%f..%f) is above of x-range (...%f)\n", left[i], right[i], x[k-1]);
			return 1;
		}
		
		/* interpolate for left  between x[j], x[j+1] and
		 *             for right between x[k], x[k+1] */
		
		wleft  = (left[i]  - x[j]) / (x[j+1] - x[j]);
		wright = (right[i] - x[k]) / (x[k+1] - x[k]);
		IFVERBOSE printf(" at <%f|%f|%f> weight %f\n", x[j],left[i], x[j+1], wleft);
		IFVERBOSE printf(" at <%f|%f|%f> weight %f\n", x[k],right[i],x[k+1], wright);
		/* y-values are samples of the values. */
		/* do a line integration between x[j] and left[i+1]
		 * and between x[k] and right[i] */
		yleft  = (y[j+1] - y[j]) * wleft  + y[j];
		yright = (y[k+1] - y[k]) * wright + y[k];		
		IFVERBOSE printf(" interpolating <%f|%f|%f>\n", y[j], yleft,  y[j+1]);
		IFVERBOSE printf(" interpolating <%f|%f|%f>\n", y[k], yright, y[k+1]);
		z[i] = yright - yleft;
	}
	return 0;
}

