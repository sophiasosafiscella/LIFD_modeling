import numpy as np
from matplotlib.pyplot import *
import emcee
import matplotlib.style
import matplotlib as mpl
rc = mpl.rc
mpl.style.use('classic')
import corner
#from pypulse.tim import Tim
import glob
import scipy.optimize as optimize
import sys
sys.path.append("/home/michael/Research/Noise Budget/source/")
import utilities as u
import psrcat
import pulsardata
PulsarData = pulsardata.PulsarData#from pulsardata import PulsarData
import jittermodels as jm
import json

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Times New Roman'],'size':14})#,'weight':'bold'})
rc('xtick',**{'labelsize':16})
rc('ytick',**{'labelsize':16})
rc('axes',**{'labelsize':18,'titlesize':18})






# http://jakevdp.github.io/blog/2015/08/07/frequentism-and-bayesianism-5-model-selection/
def compute_mcmc(lnprob,args,pinit,nwalkers=10,niter=500,threads=4):
    ndim = len(pinit)
    p0 = [pinit + 1e-4*pinit*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=args,threads=threads)
    sampler.run_mcmc(p0,niter)


    if niter <= 50000:
        burn = niter/10
    if burn > 5000:
        burn = 5000
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim)) #can also just use flatchain
    return samples



def make_triangle_plot(samples,labels,filename="triangle.png",doshow=True):
    fig = corner.corner(samples,bins=50,color='C0',smooth=0.5,plot_datapoints=False,plot_density=True,plot_contours=True,fill_contour=False,show_titles=True,labels=labels)

    fig.savefig("images/"+filename)
    if doshow:
        fig.show()
        x = raw_input()
    else:
        close()

def save_samples(samples,filename="samples.npy"):
    np.save("samples/"+filename,samples)


    
def run_constant(pd,doshow=True,weight=False,**kwargs):
    psr = pd.pulsar
    print "Running constant model: %s"%psr
    
    constant = jm.constant
    lnprob = constant.lnprob

    pinit = constant.pinit
    labels = [r"$\sigma_{\rm J,1}~[\mathrm{\mu s}]$"]

    samples = compute_mcmc(lnprob,(pd,weight),pinit,**kwargs)



    ndim = len(pinit) #redundant
    #np.save("samples.npy",samples)
    m = np.mean(samples)
    fac = np.sqrt(1800/pd.P)
    print fac


    #samples /= fac

    # https://emcee.readthedocs.io/en/latest/tutorials/line/
    for i in range(ndim):

        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        x = mcmc[1]
        xm = q[0]
        xp = q[1]
        print x,xm,xp
        print x/fac,xm/fac,xp/fac
        print
        #print map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #          zip(*np.percentile(samples[:,i], [16, 50, 84],
        #                             axis=0)))
    make_triangle_plot(samples,labels,filename="%s_constant.png"%psr,doshow=doshow)
    save_samples(samples,filename="%s_constant.npy"%psr)






def run_nband(pd,doshow=True,weight=False,**kwargs):
    psr = pd.pulsar
    print "Running N-band model: %s"%psr
    
    nband = jm.nband
    lnprob = nband.lnprob

    bands = np.array(sorted(list(set(pd.bands))))

    pinit = np.ones(len(bands))*10.0
    #np.array([10.0,10.0])
    #bands = np.array([820,1400])
    labels = map(lambda x: r"$\sigma_{\rm J,1;%i}~\mathrm{[\mu s]}$"%x,bands)



    samples = compute_mcmc(lnprob,(pd,bands,weight),pinit,**kwargs)


    ndim = len(pinit) #redundant
    m = np.mean(samples)
    fac = np.sqrt(1800/pd.P)

    # https://emcee.readthedocs.io/en/latest/tutorials/line/
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        x = mcmc[1]
        xm = q[0]
        xp = q[1]
        print x,xm,xp
        print x/fac,xm/fac,xp/fac
        print
        #print map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #          zip(*np.percentile(samples[:,i], [16, 50, 84],
        #                             axis=0)))

    make_triangle_plot(samples,labels,filename="%s_nband.png"%psr,doshow=doshow)
    save_samples(samples,filename="%s_nband.npy"%psr)



def run_powerlaw(pd,doshow=True,niter=1000,weight=False,**kwargs):
    psr = pd.pulsar
    print "Running power-law model: %s"%psr
    
    powerlaw = jm.powerlaw
    lnprob = powerlaw.lnprob

    pinit = powerlaw.pinit
    labels = [r"$\sigma_{\rm J,1;1000}~[\mathrm{\mu s}]$",r"$\alpha$"]


    samples = compute_mcmc(lnprob,(pd,weight),pinit,niter=niter,**kwargs) 


    ndim = len(pinit) #redundant
    #np.save("samples.npy",samples)
    m = np.mean(samples)
    fac = np.sqrt(1800/pd.P)

    # https://emcee.readthedocs.io/en/latest/tutorials/line/
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        x = mcmc[1]
        xm = q[0]
        xp = q[1]
        print x,xm,xp
        if i == 0:
            print x/fac,xm/fac,xp/fac
        print
        #print map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #          zip(*np.percentile(samples[:,i], [16, 50, 84],
        #                             axis=0)))

    make_triangle_plot(samples,labels,filename="%s_powerlaw.png"%psr,doshow=doshow)
    save_samples(samples,filename="%s_powerlaw.npy"%psr)


def run_powerlawconst(pd,doshow=True,niter=2000,weight=False,**kwargs):
    psr = pd.pulsar
    print "Running power-law plus constant model: %s"%psr
    
    powerlawconst = jm.powerlawconst
    lnprob = powerlawconst.lnprob

    pinit = powerlawconst.pinit
    labels = [r"$\sigma_{\rm J,1;1000}~[\mathrm{\mu s}]$",r"$\alpha$",r"$\sigma_{\rm J,1;const}~[\mathrm{\mu s}]$"]


    samples = compute_mcmc(lnprob,(pd,weight),pinit,niter=niter,**kwargs) 


    ndim = len(pinit) #redundant
    #np.save("samples.npy",samples)
    m = np.mean(samples)
    fac = np.sqrt(1800/pd.P)

    # https://emcee.readthedocs.io/en/latest/tutorials/line/
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        x = mcmc[1]
        xm = q[0]
        xp = q[1]
        print x,xm,xp
        if i == 0 or i == 2:
            print x/fac,xm/fac,xp/fac
        print
        #print map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #          zip(*np.percentile(samples[:,i], [16, 50, 84],
        #                             axis=0)))

    make_triangle_plot(samples,labels,filename="%s_powerlawconst.png"%psr,doshow=doshow)
    save_samples(samples,filename="%s_powerlawconst.npy"%psr)


def run_logpolynomial(pd,index=3,doshow=True,niter=1000,weight=False,**kwargs):
    psr = pd.pulsar
    print "Running log-polynomial model, index %i: %s"%(index,psr)
    
    logpolynomial = jm.logpolynomial
    lnprob = logpolynomial.lnprob

    pinit = np.ones(index)*10.0
    labels = map(lambda x: r"$c_{%i}$"%x,range(index))



    samples = compute_mcmc(lnprob,(pd,weight),pinit,niter=niter,**kwargs) 


    ndim = len(pinit) #redundant
    #np.save("samples.npy",samples)
    m = np.mean(samples)
    fac = np.sqrt(1800/pd.P)

    # https://emcee.readthedocs.io/en/latest/tutorials/line/
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        x = mcmc[1]
        xm = q[0]
        xp = q[1]
        print x,xm,xp
        if i == 0:
            print x/fac,xm/fac,xp/fac
        print
        #print map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #          zip(*np.percentile(samples[:,i], [16, 50, 84],
        #                             axis=0)))


    make_triangle_plot(samples,labels,filename="%s_logpolynomial%i.png"%(psr,index),doshow=doshow)
    save_samples(samples,filename="%s_logpolynomial%i.png"%(psr,index))


psr = sys.argv[1]


# Default values

residcut = 50
setbands = None
niter_constant = 1000
niter_nband = 1000
niter_powerlaw = 1000
niter_powerlawconst = 1000
niter_logpolynomial2 = 1000
niter_logpolynomial3 = 0
niter_logpolynomial4 = 0
sigma_clip = False
weight = False


#'''
with open('config.json') as FILE:
    jsondata = json.load(FILE)
    if psr in jsondata:
        for key in jsondata[psr].keys():
            #print key,str(jsondata[psr][key])
            exec("%s = %s"%(key,str(jsondata[psr][key])))
#'''

# Testing fscrunch variations
#residcut = 7.5

            
pd = PulsarData(psr,residcut=residcut,setbands=setbands)#,snrccfcut=True,snrcut=10)
if sigma_clip:
    pd.sigma_clip() #not 6?
pd.plot(filename="images/%s.png"%psr,doshow=False)


#run_powerlawconst(pd,doshow=False,weight=weight,niter=niter_powerlawconst)
#raise SystemExit



run_constant(pd,doshow=False,weight=weight,niter=niter_constant)
run_nband(pd,doshow=False,weight=weight,niter=niter_nband)
run_powerlaw(pd,doshow=False,weight=weight,niter=niter_powerlaw)
run_powerlawconst(pd,doshow=False,weight=weight,niter=niter_powerlawconst)
if niter_logpolynomial2 != 0:
    run_logpolynomial(pd,2,doshow=False,weight=weight,niter=niter_logpolynomial2)
if niter_logpolynomial3 != 0:
    run_logpolynomial(pd,3,doshow=False,weight=weight,niter=niter_logpolynomial3)
if niter_logpolynomial4 != 0:
    run_logpolynomial(pd,4,doshow=False,weight=weight,niter=niter_logpolynomial4)

