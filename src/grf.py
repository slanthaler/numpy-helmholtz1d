import numpy as np

class GRF1d:
    '''
    Implements a Gaussian random field sampler in one spatial dimension.
    The fields are produced on a periodic domain, with decay of Fourier spectrum as

    ~ exp( -(length_scale*(tau + K) )**(-alpha) )

    Args:
        length_scale: controls length scale across which there is variation in the samples
        alpha: controls the decay of Fourier coefficients
        tau: additional offset in wavenumbers to control "active" wavenumbers (|k| < tau)
    '''
    def __init__(self, length_scale=0.5, alpha=1., tau=2.):
        self.length_scale = length_scale
        self.alpha = alpha
        self.tau = tau

    def sample(self, Nsamp, Ngrid):
        # define wavenumber and coefficients
        wavenumbers = np.fft.rfftfreq(Ngrid,1/Ngrid)
        decay_coeff = np.exp( -(self.length_scale * (np.abs(wavenumbers)+self.tau))**self.alpha )
        decay_coeff /= np.linalg.norm(decay_coeff)
        shape = (Nsamp, len(wavenumbers))
        rand_coeff = np.random.randn(*shape) + 1j*np.random.randn(*shape)

        # Fourier coefficients of u
        uhat = decay_coeff.reshape(1,-1) * rand_coeff

        # transform back via (real) inverse FT
        u = np.fft.irfft(uhat, norm='forward')

        return u

    @staticmethod
    def grid(Ngrid, periodic=True):
        if periodic:
            return np.linspace(0,1,Ngrid+1)[:-1] # periodic grid
        else:
            return np.linspace(0,1,Ngrid) # non-periodic grid
