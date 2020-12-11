import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.special import gamma
from math import exp, pi, sqrt


#Intializing parameters
def init(N):
    X = np.random.normal(0, 1, N)
    X_mean = X.mean()
    a0 = 0
    b0 = 0
    mu0 = 0
    lambda0 = 0
    return X, X_mean, a0, b0, mu0, lambda0

#Calculating true parameters
def trueParameters(N, X, X_mean, a0, b0, mu0, lambda0):
    aTrue = a0 + (N) / 2
    bTrue = b0 + (1 / 2)*sum((X - X_mean)**2) + (lambda0*N
            *(X_mean - mu0)**2)/(2*(lambda0 + N))
    muTrue = (lambda0*mu0 + N*X_mean)/(lambda0 + N)
    lambdaTrue = lambda0 + N
    return muTrue, lambdaTrue, aTrue, bTrue

#Calculating iterative parameters and initalizing an "old" lambda
# value needed for threshold comparison
def nParameters(N, X_mean, a0, mu0, lambda0):
    muN =(lambda0*mu0 + N*X_mean)/(lambda0 + N)
    lambdaN = 3
    aN = a0 + N/2
    bN = 3
    lambdaOld = lambdaN
    return muN, lambdaN, aN, bN, lambdaOld

#Updating parameters lambdaN and bN
def updateParameters(X, N, mu0, lambda0, a0, b0, muN, lambdaN, aN, bN):
    Emu = muN
    Emuu = (1/lambdaN) + muN**2
    E_tau = aN/bN
    lambdaN = (lambda0 + N)*E_tau
    bLeft = sum(X**2) - 2*sum(X)*Emu + N*Emuu
    bRight = lambda0*(Emuu - 2*mu0*Emu + mu0**2)
    bN = b0 +(1/2)*(bLeft+bRight)
    return lambdaN, bN

#Calculating true distribution
def normalGamma(muTrue, lambdaTrue, aTrue, bTrue, mu, tau):
    nG = (bTrue**aTrue)*sqrt(lambdaTrue)/(gamma(aTrue)*sqrt(2*pi))\
         *tau**(aTrue - (1/2))*np.exp(-bTrue*tau)*np.exp(- (1/2)\
         *lambdaTrue*np.dot(tau,((mu-muTrue)**2).T))
    return nG

#Calculating approximate distribution
def approximateDistribution(muN, lambdaN, aN, bN, mu, tau):
    D_mu = sqrt(lambdaN/(2*pi))*exp(-(1/2)*np.dot(lambdaN,((mu-muN)**2).T))
    D_tau = (1/gamma(aN))*(bN**aN)*(tau**(aN-1))*exp(- bN*tau)
    D = D_mu*D_tau
    return D

#Calculating PDF for mu
def muPdf(x, mu, sigma):
    return stats.norm.pdf(x, mu, 1/sigma )

#Calculating PDF for tau
def tauPdf (x , a , b):
    return stats.gamma.pdf(x, a, loc=0, scale =1/b)

#Plot PDFs
def plot(muN, lambdaN, aN, bN, muTrue, lambdaTrue, aTrue, bTrue,
         mu, tau, iter):
    _mu, _tau = np.meshgrid(mu, tau, indexing ='ij')
    approximateDist = np.zeros_like(_mu)
    for i in range (approximateDist.shape[0]) :
        for j in range (approximateDist.shape[1]) :
            approximateDist[i][j] = muPdf(mu[i], muN, sqrt(lambdaN))\
                                    *tauPdf(tau[j], aN, bN)
    plt.plot()
    plt.title('Iteration nr: ' + str(iter))
    plt.contour(_mu, _tau, approximateDist, colors ="blue")
    plt.xlabel ("Mu")
    plt.ylabel ("Tau")
    trueDist = np.zeros_like(_mu)
    for i in range(trueDist.shape[0]):
        for j in range(trueDist.shape[1]):
            trueDist[i][j] = normalGamma(muTrue, lambdaTrue, aTrue,
                             bTrue, mu[i], tau[j])
    plt.contour(_mu, _tau, trueDist, colors ="green")
    plt.show()
    plt.close()
    return


def main():
    np.random.seed(0)
    #Setting max iterations and threshold
    iter = 10
    epsilon = 10 ** (-2)
    #Setting number of datapoints
    N = 100
    mu = np.linspace(-1.5, 1.5, 100)
    tau = np.linspace(0, 3, 100)
    X, X_mean, a0, b0, mu0, lambda0 = init(N)
    muTrue, lambdaTrue, aTrue, bTrue = trueParameters(N, X, X_mean,
                                        a0, b0, mu0, lambda0)
    muN, lambdaN, aN, bN, lambdaOld = nParameters(N, X_mean, a0, mu0,
                                                  lambda0)
    for i in range(iter):
        lambdaN, bN = updateParameters(X, N, mu0, lambda0, a0, b0,
                                       muN, lambdaN, aN, bN)
        if(abs(lambdaN-lambdaOld) < epsilon):
            plot(muN, lambdaN, aN, bN, muTrue, lambdaTrue, aTrue,
                 bTrue, mu, tau, i)
            break
        else:
            plot(muN, lambdaN, aN, bN, muTrue, lambdaTrue, aTrue,
                 bTrue, mu, tau, i)
        lambdaOld = lambdaN
    print(muN, lambdaN, aN, bN)

if __name__ == "__main__":
    main()
