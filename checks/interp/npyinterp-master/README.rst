--------------------------------------------------------------------------
Fast interpolation/integration for monotonically increasing numpy arrays
--------------------------------------------------------------------------

Integrates from a to b, given a array of data points, by using piecewise
linear interpolation. Very fast.

Description
------------

You know numpy.interp(x, xp, yp)? It interpolates x using the arrays xp 
(coordinates, increasing) and yp (values). It does so via a binary search to 
find the closest coordinate to x in xp.

If you have a large array of x values this can be slow because numpy.interp
does a binary search every time. This tiny library assumes that the x values
are ordered, and thus continues its search in the vicinity of the last lookup.

This makes this interpolation very fast.

This library (just 1 C function) actually does not just interpolate, but 
integrates bins, which are defined by a lower bin border and an upper bin 
border. The integration is linear piecewise. 

How to install
---------------

 $ make
 
And point LD_LIBRARY_PATH to the directory containing it.

How to use
-----------------------

See monointerp.py for calling from Python



