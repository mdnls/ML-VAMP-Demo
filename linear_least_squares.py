from lib.mlvamp import LayerEstimator, Linear, GaussianPrior
import numpy as np

if __name__ == "__main__":
    s1 = 1          # ratio of measurements to ambient dimension
    snr = 10        # ratio of variance of signal to that of noise
    n = 500
    m = int(s1*n)
    A = np.random.normal(size=(m, n), scale=1/np.sqrt(m))
    z = np.random.normal(size=(n,), scale=1)
    eta = np.random.normal(size=(m,), scale=1/np.sqrt(snr))

    y = A @ z + eta

    P = GaussianPrior(mode="map", dim=n, mean=np.zeros((n,)), stddev=1)
    L = Linear(mode="map", A=A, b=np.zeros((m,)), in_dim=(n,), out_dim=(m,),
                        measurements=eta, measurement_prec=snr) # prec is 1/var

    P.register_neighbors(P.incoming, L)
    L.register_neighbors(P, L.outgoing)
    P.parameter_init(n, n, gm_init=1)
    L.parameter_init(n, m, gm_init=1)

    '''
    Todo
    1. Figure out the order of forward and backward passes. 
    2. That order should have handling for the beginning and end pieces which only have forward and backward passes resp. 
    3. Index 0 is the prior. Index 1 is the only layer. 
        - either theta does a backward pass pulling from linear, or linear does a backward pass overwriting theta. 
    '''
    for i in range(10):
        P.step(dir="fwd")
        P.step(dir="bck")
        z_error = np.linalg.norm(P.params['z-'] - z)
        measurement_error = np.linalg.norm(A @ P.params['z-'] - y)
        print(f"Iter {i} -- MSE {z_error} -- Measurement error {measurement_error}")
