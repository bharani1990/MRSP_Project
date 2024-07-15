import torch


def f_SP_dB_torch(maxfreq, nfilts):
    # usage: spreadingfunctionmatdB=f_SP_dB(maxfreq,nfilts)
    # computes the spreading function protoype, in the Bark scale.
    # Arguments: maxfreq: half the sampling freqency
    # nfilts: Number of subbands in the Bark domain, for instance 64
    # upper end of our Bark scale:22 Bark at 16 kHz
    maxbark = hz2bark_torch(maxfreq)
    # Number of our Bark scale bands over this range: nfilts=64
    spreadingfunctionBarkdB = torch.zeros(2*nfilts)
    # Spreading function prototype, "nfilts" bands for lower slope
    spreadingfunctionBarkdB[0:nfilts] = torch.linspace(
        -maxbark*27, -8, nfilts)-23.5
    # "nfilts" bands for upper slope:
    spreadingfunctionBarkdB[nfilts:2 *
                            nfilts] = torch.linspace(0, -maxbark*12.0, nfilts)-23.5
    return spreadingfunctionBarkdB


def spreadingfunctionmat_torch(spreadingfunctionBarkdB, alpha, nfilts):
    # Turns the spreading prototype function into a matrix of shifted versions.
    # Convert from dB to "voltage" and include alpha exponent
    # nfilts: Number of subbands in the Bark domain, for instance 64
    spreadingfunctionBarkVoltage = 10.0**(
        spreadingfunctionBarkdB/20.0*alpha)
    # Spreading functions for all bark scale bands in a matrix:
    spreadingfuncmatrix = torch.zeros((nfilts, nfilts))
    for k in range(nfilts):
        spreadingfuncmatrix[k, :] = spreadingfunctionBarkVoltage[(
            nfilts-k):(2*nfilts-k)]
    return spreadingfuncmatrix


def maskingThresholdBark_torch(mXbark, spreadingfuncmatrix, alpha, fs, nfilts):
    # Computes the masking threshold on the Bark scale with non-linear superposition
    # usage: mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha)
    # Arg: mXbark: magnitude of FFT spectrum, on the Bark scale
    # spreadingfuncmatrix: spreading function matrix from function spreadingfunctionmat
    # alpha: exponent for non-linear superposition (eg. 0.6),
    # fs: sampling freq., nfilts: number of Bark subbands
    # nfilts: Number of subbands in the Bark domain, for instance 64
    # Returns: mTbark: the resulting Masking Threshold on the Bark scale

    # Compute the non-linear superposition:
    mTbark = torch.matmul(mXbark**alpha, spreadingfuncmatrix**alpha)
    # apply the inverse exponent to the result:
    mTbark = mTbark**(1.0/alpha)
    # Threshold in quiet:
    maxfreq = fs/2.0
    maxbark = hz2bark_torch(maxfreq)
    step_bark = maxbark/(nfilts-1)
    barks = torch.arange(0, nfilts)*step_bark
    # convert the bark subband frequencies to Hz:
    f = bark2hz_torch(barks)+1e-6
    # Threshold of quiet in the Bark subbands in dB:
    LTQ = torch.clip((3.64*(f/1000.)**-0.8 - 6.5*torch.exp(-0.6*(f/1000.-3.3)**2.)
                      + 1e-3*((f/1000.)**4.)), -20, 120)
    # Maximum of spreading functions and hearing threshold in quiet:
    a = mTbark
    b = 10.0**((LTQ-60)/20)
    mTbark = torch.max(a, b)
    return mTbark


def hz2bark_torch(f):
    """ Usage: Bark=hz2bark(f)
          f    : (ndarray)    Array containing frequencies in Hz.
      Returns  :
          Brk  : (ndarray)    Array containing Bark scaled values.
      """
    if not torch.is_tensor(f):
        f = torch.tensor(f)

    Brk = 6. * torch.arcsinh(f/600.)
    return Brk


def bark2hz_torch(Brk):
    """ Usage:
      Hz=bark2hs(Brk)
      Args     :
          Brk  : (ndarray)    Array containing Bark scaled values.
      Returns  :
          Fhz  : (ndarray)    Array containing frequencies in Hz.
      """
    if not torch.is_tensor(Brk):
        Brk = torch.tensor(Brk)
    Fhz = 600. * torch.sinh(Brk/6.)
    return Fhz


def mapping2barkmat_torch(fs, nfilts, nfft):
    # Constructing mapping matrix W which has 1's for each Bark subband, and 0's else
    # usage: W=mapping2barkmat(fs, nfilts,nfft)
    # arguments: fs: sampling frequency
    # nfilts: number of subbands in Bark domain
    # nfft: number of subbands in fft
    # upper end of our Bark scale:22 Bark at 16 kHz
    maxbark = hz2bark_torch(fs/2)
    nfreqs = nfft/2
    step_bark = maxbark/(nfilts-1)
    binbark = hz2bark_torch(
        torch.linspace(0, (nfft/2), (nfft//2)+1)*fs/nfft)
    W = torch.zeros((nfilts, nfft))
    for i in range(nfilts):
        W[i, 0:int(nfft/2)+1] = (torch.round(binbark/step_bark) == i)
    return W


def mapping2bark_torch(mX, W, nfft):
    # Maps (warps) magnitude spectrum vector mX from DFT to the Bark scale
    # arguments: mX: magnitude spectrum from fft
    # W: mapping matrix from function mapping2barkmat
    # nfft: : number of subbands in fft
    # returns: mXbark, magnitude mapped to the Bark scale
    nfreqs = int(nfft/2)
    # Here is the actual mapping, suming up powers and conv. back to Voltages:
    mXbark = (torch.matmul(
        torch.abs(mX[:nfreqs])**2.0, W[:, :nfreqs].T))**(0.5)
    return mXbark


def mappingfrombarkmat_torch(W, nfft):
    # Constructing inverse mapping matrix W_inv from matrix W for mapping back from bark scale
    # usuage: W_inv=mappingfrombarkmat(Wnfft)
    # argument: W: mapping matrix from function mapping2barkmat
    # nfft: : number of subbands in fft
    nfreqs = int(nfft/2)
    W_inv = torch.matmul(torch.diag(
        (1.0/(torch.sum(W, 1)+1e-6))**0.5), W[:, 0:nfreqs + 1]).T
    return W_inv

# -------------------


def mappingfrombark_torch(mTbark, W_inv, nfft):
    # usage: mT=mappingfrombark(mTbark,W_inv,nfft)
    # Maps (warps) magnitude spectrum vector mTbark in the Bark scale
    # back to the linear scale
    # arguments:
    # mTbark: masking threshold in the Bark domain
    # W_inv : inverse mapping matrix W_inv from matrix W for mapping back from bark scale
    # nfft: : number of subbands in fft
    # returns: mT, masking threshold in the linear scale
    nfreqs = int(nfft/2)
    mT = torch.matmul(mTbark, W_inv[:, :nfreqs].T.float())
    return mT

def psyacthresh_torch(ys, fs):
    # input: ys: 2d array of sound STFT (from a mono signal, shape N+1,M)
    # fs: sampling frequency in samples per second
    # returns: mT, the masking threshold in N+1 subbands for the M blocks (shape N+1,M)

    maxfreq = fs/2
    alpha = 0.8  # Exponent for non-linear superposition of spreading functions
    nfilts = 64  # number of subbands in the bark domain
    # M=len(snd)//nfft
    M = ys.shape[1]
    # N=nfft//2
    N = ys.shape[0]-1
    nfft = 2*N

    W = mapping2barkmat_torch(fs, nfilts, nfft)
    W_inv = mappingfrombarkmat_torch(W, nfft)
    spreadingfunctionBarkdB = f_SP_dB_torch(maxfreq, nfilts)
    # maxbark=hz2bark(maxfreq)
    # bark=np.linspace(0,maxbark,nfilts)
    spreadingfuncmatrix = spreadingfunctionmat_torch(
        spreadingfunctionBarkdB, alpha, nfilts)
    # Computing the masking threshold in each block of nfft samples:
    mT = torch.zeros((N+1, M))
    for m in range(M):  # M: number of blocks
        # mX=np.abs(np.fft.fft(snd[m*nfft+np.arange(2048)],norm='ortho'))[0:1025]
        mX = torch.abs(ys[:, m])
        mXbark = mapping2bark_torch(mX, W, nfft)
        # Compute the masking threshold in the Bark domain:
        mTbark = maskingThresholdBark_torch(
            mXbark, spreadingfuncmatrix, alpha, fs, nfilts)
        # Massking threshold in the original frequency domain
        mT[:, m] = mappingfrombark_torch(mTbark, W_inv, nfft)

    return mT  # the masking threshold in N+1 subbands for the M blocks


def percloss(orig, modified, fs):
    # computes the perceptually weighted distance between the original (orig) and modified audio signals,
    # with sampling rate fs. The psycho-acoustic threshold is computed from orig, hence it is not commutative.
    # returns: ploss, the perceptual loss value, the mean squarred difference of the two spectra, normalized to the masking threshold of the orig.
    # Gerald Schuller, September 2023

    nfft = 2048  # number of fft subbands
    N = nfft//2

    # print("orig.shape=", orig.shape)
    
    # origsys.shape= freq.bin, channel, block
    if len(orig.shape) == 2:  # multichannel
        chan = orig.shape[1]
        for c in range(chan):
            origys = torch.stft(orig[:,c], n_fft=2*N, hop_length=2 *
                        N//2, return_complex=True, normalized=True, window=torch.hann_window(2*N))
            if c == 0:  # initialize masking threshold tensor mT
                mT0 = psyacthresh_torch(origys[:, :], fs)
                rows, cols = mT0.shape
                mT = torch.zeros((rows, chan, cols))
                mT[:, 0, :] = mT0
            else:
                mT[:, c, :] = psyacthresh_torch(origys[:, :], fs)
    else:
        chan = 1
        origys = torch.stft(orig, n_fft=2*N, hop_length=2 *
                        N//2, return_complex=True, normalized=True, window=torch.hann_window(2*N))
        mT = psyacthresh_torch(origys, fs)
    """
    plt.plot(20*np.log10(np.abs(origys[:,0,400])+1e-6))
    plt.plot(20*np.log10(mT[:,0,400]+1e-6))
    plt.legend(('Original spectrum','Masking threshold'))
    plt.title("Spectrum over bins")
    """
    # print("origys.shape=",origys.shape, "mT.shape=",mT.shape)

    modifiedys = torch.stft(
        modified, n_fft=2*N, hop_length=2*N//2, return_complex=True, normalized=True, window=torch.hann_window(2*N))

    # normalized diff. spectrum:
    normdiffspec = torch.abs((origys-modifiedys)/mT)
    # Plot difference spectrum, normalized to masking threshold:
    """
    plt.plot(20*np.log10(normdiffspec[:,0,400])+1e-6)
    plt.title("normalized diff. spectrum")
    plt.show()
    """
    ploss = torch.mean(normdiffspec**2)
    return ploss
