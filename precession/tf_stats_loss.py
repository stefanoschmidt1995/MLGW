import tensorflow as tf

##########################################################
#       STATISTICAL DISTANCES(LOSSES) IN TENSORFLOW      #
##########################################################

## Statistial Distances for 1D weight distributions
## Inspired by Scipy.Stats Statistial Distances for 1D
## Tensorflow Version, to make a valid Loss
## The code here is a tf version from https://github.com/TakaraResearch/Pytorch-1D-Wasserstein-Statistical-Loss

def tf_wasserstein_loss(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(tf_cdf_loss(tensor_a,tensor_b,p=1))

def tf_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*tf_cdf_loss(tensor_a,tensor_b,p=2))

def tf_cdf_loss(tensor_a,tensor_b,p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (tf.math.reduce_sum(tensor_a, axis=-1, keepdims=True) + 1e-14)
    tensor_b = tensor_b / (tf.math.reduce_sum(tensor_b, axis=-1, keepdims=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = tf.math.cumsum(tensor_a,axis=-1)
    cdf_tensor_b = tf.math.cumsum(tensor_b,axis=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = tf.math.reduce_sum(tf.math.abs((cdf_tensor_a-cdf_tensor_b)),axis=-1)
    elif p == 2:
        cdf_distance = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow((cdf_tensor_a-cdf_tensor_b),2),axis=-1))
    else:
        cdf_distance = tf.math.pow(tf.math.reduce_sum(tf.pow(tf.math.abs(cdf_tensor_a-cdf_tensor_b),p),axis=-1),1/p)
    print("Ciao", tensor_a.shape, cdf_tensor_a.shape)
    import matplotlib.pyplot as plt
    plt.plot(tensor_a.numpy())
    plt.plot(tensor_b.numpy())
    plt.show()
    

    cdf_loss = tf.math.reduce_mean(cdf_distance)
    return cdf_loss

