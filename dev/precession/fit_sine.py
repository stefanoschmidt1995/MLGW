import numpy as np
import pylab as plt
import scipy.optimize as optimize

# This is your target function
def sineFit(t, a, f, p):
    return a * np.sin(2.0*np.pi*f*t + p)

# This is our "error" function
def err_func(p0, X, Y, target_function):
    err = ((Y - target_function(X, *p0))**2).sum()
    return err


# Try out different parameters, sometimes the random guess works
# sometimes it fails. The FFT solution should always work for this problem
inital_args = np.random.random(3)

X = np.linspace(0, 10, 1000)
Y = sineFit(X, *inital_args)

# Use a random inital guess
inital_guess = np.random.random(3)

# Fit
sol = optimize.fmin(err_func, inital_guess, args=(X,Y,sineFit))

# Plot the fit
Y2 = sineFit(X, *sol)
plt.figure(figsize=(15,10))
plt.subplot(211)
plt.title("Random Inital Guess: Final Parameters: %s"%sol)
plt.plot(X,Y)
plt.plot(X,Y2,'r',alpha=.5,lw=10)

# Use an improved "fft" guess for the frequency
# this will be the max in k-space
timestep = X[1]-X[0]
guess_k = np.argmax( np.fft.rfft(Y) )
guess_f = np.fft.fftfreq(X.size, timestep)[guess_k]
inital_guess[1] = guess_f 

# Guess the amplitiude by taking the max of the absolute values
inital_guess[0] = np.abs(Y).max()

sol = optimize.fmin(err_func, inital_guess, args=(X,Y,sineFit))
Y2 = sineFit(X, *sol)

plt.subplot(212)
plt.title("FFT Guess          : Final Parameters: %s"%sol)
plt.plot(X,Y)
plt.plot(X,Y2,'r',alpha=.5,lw=10)
plt.show()
