"""

% Copyright:	Andrew L. Ferguson, UIUC 
% Last updated:	2 Jan 2016

% SYNOPSIS
%
% code to perform plotting of: 	(i)   {1,2,3}-dimensional maximum a posteriori (MAP) free energy surfaces (FES) computed by Bayesian inference of biased umbrella sampling trajectories in umbrella variables psi 
%								(ii)  uncertainty estimates in MAP FES estimated by Metropolis-Hastings (MH) sampling of Bayes posterior distribution 
% 								(iii) likelihood trajectory along MH sampling path to assess convergence of MH sampling 

% INPUTS
%
% f__hist_binCenters  		- [str] [dim x M_k float] path to text file containing dim rows specifying M_k k=1..dim centers of the rectilinear histogram bins in each dimension used to construct dim-dimensional histograms constituting f__pdf_MAP/MH and f__betaF_MAP/MH 
% f__hist_binWidths   		- [str] [dim x M_k float] path to text file containing dim rows specifying M_k k=1..dim widths of the rectilinear histogram bins in each dimension used to construct dim-dimensional histograms constituting f__pdf_MAP/MH and f__betaF_MAP/MH 
% f__pdf_MAP          		- [str] [1 x M float] path to text file containing MAP estimate of unbiased probability density function pdf_l over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) 
% f__betaF_MAP        		- [str] [1 x M float] path to text file containing MAP estimate of unbiased free energy surface beta*F_l = -ln(p(psi)/binVolume) + const. over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) 
% f__f_MAP            		- [str] [1 x S float] path to text file containing MAP estimates of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations 
% f__pdf_MH           		- [str] [nSamples_MH x M float] path to text file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased probability density function pdf_l over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) 
% f__betaF_MH         		- [str] [nSamples_MH x M float] path to text file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased free energy surface beta*F_l = -ln(p(psi)/binVolume) + const. over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) 
% f__f_MH             		- [str] [nSamples_MH x S float] path to text file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations 
% f__logL_MH          		- [str] [nSamples_MH x 1 float] path to text file containing log of Bayes posterior up to an additive constant over the nSamples_MH Metropolis-Hastings samples from the posterior 
% f__step_MH          		- [str] [nSamples_MH x 1 int] path to text file containing Metropolis-Hastings step associated with each of the nSamples_MH Metropolis-Hastings samples from the posterior 

% OUTPUTS
%
% logL.jpg/eps				- trajectory of the log Bayes posterior up to an additive constant over the nSamples_MH Metropolis-Hastings samples from the Bayesian posterior 
% f.jpg/eps					- MAP estimates of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i for i=1..S biased simulations, overlaid with traces of MH samples of f_i 
%
% 1-dimension:
%
% pdf.jpg/eps				- MAP estimate of probability density function as a function of psi 
% pdf_traces.jpg/eps		- MAP estimate of probability density function as a function of psi overlaid with traces of MH samples 
% pdf_eb_limits.jpg/eps		- MAP estimate of probability density function as a function of psi overlaid with the MAP estimates plus and minus the standard deviation of the ensemble of MH samples 
% pdf_eb_stdev.jpg/eps		- standard deviation of the probability density function of the ensemble of MH samples as a function of psi 
% pdf_eb_bars.jpg/eps		- MAP estimate of probability density function as a function of psi with error bars denoting the standard deviation of the ensemble of MH samples 
% betaF.jpg/eps				- MAP estimate of beta*F = -ln(p(psi)/binVolume) + const. as a function of psi 
% betaF_traces.jpg/eps		- MAP estimate of beta*F = -ln(p(psi)/binVolume) + const. as a function of psi overlaid with traces of MH samples 
% betaF_eb_limits.jpg/eps	- MAP estimate of beta*F = -ln(p(psi)/binVolume) + const. as a function of psi overlaid with the MAP estimates plus and minus the standard deviation of the ensemble of MH samples 
% betaF_eb_stdev.jpg/eps	- standard deviation of the beta*F = -ln(p(psi)/binVolume) + const. of the ensemble of MH samples as a function of psi 
% betaF_eb_bars.jpg/eps		- MAP estimate of beta*F = -ln(p(psi)/binVolume) + const. as a function of psi with error bars denoting the standard deviation of the ensemble of MH samples 
% betaF_eb_dist.jpg/eps		- MAP estimate of beta*F = -ln(p(psi)/binVolume) + const. as a function of psi overlaid with the probability density function of the MH ensemble at each MAP data point 
%
% 2-dimensions:
%
% pdf.jpg/eps				- MAP estimate of probability density function as a function of psi 
% pdf_traces.jpg/eps		- MAP estimate of probability density function as a function of psi overlaid with traces of MH samples 
% pdf_eb_limits.jpg/eps		- MAP estimate of probability density function as a function of psi overlaid with the MAP estimates plus and minus the standard deviation of the ensemble of MH samples 
% pdf_eb_stdev.jpg/eps		- standard deviation of the probability density function of the ensemble of MH samples as a function of psi 
% betaF.jpg/eps				- MAP estimate of beta*F = -ln(p(psi)/binVolume) + const. as a function of psi 
% betaF_traces.jpg/eps		- MAP estimate of beta*F = -ln(p(psi)/binVolume) + const. as a function of psi overlaid with traces of MH samples 
% betaF_eb_limits.jpg/eps	- MAP estimate of beta*F = -ln(p(psi)/binVolume) + const. as a function of psi overlaid with the MAP estimates plus and minus the standard deviation of the ensemble of MH samples 
% betaF_eb_stdev.jpg/eps	- standard deviation of the beta*F = -ln(p(psi)/binVolume) + const. of the ensemble of MH samples as a function of psi 
%
% 3-dimensions:
%
% pdf.jpg					- MAP estimate of probability density function as a function of psi 
% pdf_eb_stdev.jpg			- standard deviation of the probability density function of the ensemble of MH samples as a function of psi 
% betaF.jpg					- MAP estimate of beta*F = -ln(p(psi)/binVolume) + const. as a function of psi 
% betaF_eb_stdev.jpg		- standard deviation of the beta*F = -ln(p(psi)/binVolume) + const. of the ensemble of MH samples as a function of psi 

"""

## imports
import os, re, sys, time
import random, math

import numpy as np
import numpy.matlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

## classes

## methods

# usage
def _usage():
	print "USAGE: %s f__hist_binCenters f__hist_binWidths f__pdf_MAP f__betaF_MAP f__f_MAP f__pdf_MH f__betaF_MH f__f_MH f__logL_MH f__step_MH" % sys.argv[0]
	print "       f__hist_binCenters  - [str] [dim x M_k float] path to text file containing dim rows specifying M_k k=1..dim centers of the rectilinear histogram bins in each dimension used to construct dim-dimensional histograms constituting f__pdf_MAP/MH and f__betaF_MAP/MH "
	print "       f__hist_binWidths   - [str] [dim x M_k float] path to text file containing dim rows specifying M_k k=1..dim widths of the rectilinear histogram bins in each dimension used to construct dim-dimensional histograms constituting f__pdf_MAP/MH and f__betaF_MAP/MH "
	print "       f__pdf_MAP          - [str] [1 x M float] path to text file containing MAP estimate of unbiased probability density function pdf_l over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) "
	print "       f__betaF_MAP        - [str] [1 x M float] path to text file containing MAP estimate of unbiased free energy surface beta*F_l = -ln(p(psi)/binVolume) + const. over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) "
	print "       f__f_MAP            - [str] [1 x S float] path to text file containing MAP estimates of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations "
	print "       f__pdf_MH           - [str] [nSamples_MH x M float] path to text file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased probability density function pdf_l over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) "
	print "       f__betaF_MH         - [str] [nSamples_MH x M float] path to text file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased free energy surface beta*F_l = -ln(p(psi)/binVolume) + const. over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) "
	print "       f__f_MH             - [str] [nSamples_MH x S float] path to text file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations "
	print "       f__logL_MH          - [str] [nSamples_MH x 1 float] path to text file containing log of Bayes posterior up to an additive constant over the nSamples_MH Metropolis-Hastings samples from the posterior "
	print "       f__step_MH          - [str] [nSamples_MH x 1 int] path to text file containing Metropolis-Hastings step associated with each of the nSamples_MH Metropolis-Hastings samples from the posterior "
	print "       OR"
	print "USAGE: %s" % sys.argv[0]
	print "       \-> to accept default arguments:"
	print "       f__hist_binCenters  = hist_binCenters.txt"
	print "       f__hist_binWidths   = hist_binWidths.txt"
	print "       f__pdf_MAP          = pdf_MAP.txt"
	print "       f__betaF_MAP        = betaF_MAP.txt"
	print "       f__f_MAP            = f_MAP.txt"
	print "       f__pdf_MH           = pdf_MH.txt"
	print "       f__betaF_MH         = betaF_MH.txt"
	print "       f__f_MH             = f_MH.txt"
	print "       f__logL_MH          = logL_MH.txt"
	print "       f__step_MH          = step_MH.txt"
	
def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))



## main

# parameters
linewidth = 0.5

# loading inputs
if len(sys.argv) == 1:
	f__hist_binCenters = 'hist_binCenters.txt'
	f__hist_binWidths = 'hist_binWidths.txt'
	f__pdf_MAP = 'pdf_MAP.txt'
	f__betaF_MAP = 'betaF_MAP.txt'
	f__f_MAP = 'f_MAP.txt'
	f__pdf_MH = 'pdf_MH.txt'
	f__betaF_MH = 'betaF_MH.txt'
	f__f_MH = 'f_MH.txt'
	f__logL_MH = 'logL_MH.txt'
	f__step_MH = 'step_MH.txt'
elif len(sys.argv) == 11:
	f__hist_binCenters = str(sys.argv[1])
	f__hist_binWidths = str(sys.argv[2])
	f__pdf_MAP = str(sys.argv[3])
	f__betaF_MAP = str(sys.argv[4])
	f__f_MAP = str(sys.argv[5])
	f__pdf_MH = str(sys.argv[6])
	f__betaF_MH = str(sys.argv[7])
	f__f_MH = str(sys.argv[8])
	f__logL_MH = str(sys.argv[9])
	f__step_MH = str(sys.argv[10])
else:
	_usage()
	sys.exit(-1)

# - printing args to screen
print("")
print("f__hist_binCenters = %s" % (f__hist_binCenters))
print("f__hist_binWidths = %s" % (f__hist_binWidths))
print("f__pdf_MAP = %s" % (f__pdf_MAP))
print("f__betaF_MAP = %s" % (f__betaF_MAP))
print("f__f_MAP = %s" % (f__f_MAP))
print("f__pdf_MH = %s" % (f__pdf_MH))
print("f__betaF_MH = %s" % (f__betaF_MH))
print("f__f_MH = %s" % (f__f_MH))
print("f__logL_MH = %s" % (f__logL_MH))
print("f__step_MH = %s" % (f__step_MH))
print("")


# loading data
print("Loading data...")

binC = []
with open(f__hist_binCenters,'r') as fin:
	for line in fin:
		binC.append(line.strip().split())
binC = [[float(y) for y in x] for x in binC]
binC = np.array(binC)

binW = []
with open(f__hist_binWidths,'r') as fin:
	for line in fin:
		binW.append(line.strip().split())
binW = [[float(y) for y in x] for x in binW]
binW = np.array(binW)

fin = open(f__pdf_MAP,'r')
line = fin.readline()
pdf_MAP = line.strip().split()
pdf_MAP = [float(x) for x in pdf_MAP]
pdf_MAP = np.array(pdf_MAP)
fin.close()

fin = open(f__betaF_MAP,'r')
line = fin.readline()
betaF_MAP = line.strip().split()
betaF_MAP = [float(x) for x in betaF_MAP]
betaF_MAP = np.array(betaF_MAP)
fin.close()

fin = open(f__f_MAP,'r')
line = fin.readline()
f_MAP = line.strip().split()
f_MAP = [float(x) for x in f_MAP]
f_MAP = np.array(f_MAP)
fin.close()

pdf_MH = []
with open(f__pdf_MH,'r') as fin:
	for line in fin:
		pdf_MH.append(line.strip().split())
pdf_MH = [[float(y) for y in x] for x in pdf_MH]
pdf_MH = np.array(pdf_MH)

betaF_MH = []
with open(f__betaF_MH,'r') as fin:
	for line in fin:
		betaF_MH.append(line.strip().split())
betaF_MH = [[float(y) for y in x] for x in betaF_MH]
betaF_MH = np.array(betaF_MH)

f_MH = []
with open(f__f_MH,'r') as fin:
	for line in fin:
		f_MH.append(line.strip().split())
f_MH = [[float(y) for y in x] for x in f_MH]
f_MH = np.array(f_MH)

logL_MH = []
with open(f__logL_MH,'r') as fin:
	for line in fin:
		logL_MH.append(line.strip())
logL_MH = [float(x) for x in logL_MH]
logL_MH = np.array(logL_MH)

step_MH = []
with open(f__step_MH,'r') as fin:
	for line in fin:
		step_MH.append(line.strip())
step_MH = [float(x) for x in step_MH]
step_MH = np.array(step_MH)

print("DONE!\n\n")


# post-processing and error checking
print("Error checking data import...")

dim = len(binC)
if len(binW) != dim:
	print("\nERROR - Dimensionality of %s and %s are incompatible" % (f__hist_binCenters,f__hist_binWidths))
	sys.exit(-1)

M_k = np.zeros(dim, dtype=np.uint64)
for d in range(0,dim):
	M_k[d] = len(binC[d])
	if len(binW[d]) != M_k[d]:
		print("\nERROR - Number of bins in dimension %d of %s and %s are incompatible" % (d,f__hist_binCenters,f__hist_binWidths))
		sys.exit(-1)

M = np.prod(M_k)
if pdf_MAP.shape[0] != M:
	print("\nERROR - Number of bins in %s is incompatible with %s and %s" % (f__pdf_MAP,f__hist_binCenters,f__hist_binWidths))
	sys.exit(-1)
if betaF_MAP.shape[0] != M:
	print("\nERROR - Number of bins in %s is incompatible with %s and %s" % (f__betaF_MAP,f__hist_binCenters,f__hist_binWidths))
	sys.exit(-1)

S = f_MAP.shape[0]

nSamples_MH = pdf_MH.shape[0]
if pdf_MH.shape[1] != M:
	print("\nERROR - Number of bins in %s is incompatible with %s and %s" % (f__pdf_MH,f__hist_binCenters,f__hist_binWidths))
	sys.exit(-1)
if betaF_MH.shape[0] != nSamples_MH:
	print("\nERROR - Number of rows (i.e., M-H samples) in %s is incommensurate with that in %s" % (f__betaF_MH,f__pdf_MH))
	sys.exit(-1)
if betaF_MH.shape[1] != M:
	print("\nERROR - Number of bins in %s is incompatible with %s and %s" % (f__betaF_MH,f__hist_binCenters,f__hist_binWidths))
	sys.exit(-1)
if f_MH.shape[0] != nSamples_MH:
	print("\nERROR - Number of rows (i.e., M-H samples) in %s is incommensurate with that in %s" % (f__f_MH,f__pdf_MH))
	sys.exit(-1)
if f_MH.shape[1] != S:
	print("\nERROR - Number of columns (i.e., simulations) in %s is incompatible with %s" % (f__f_MH,f__f_MAP))
	sys.exit(-1)
if logL_MH.shape[0] != nSamples_MH:
	print("\nERROR - Number of rows (i.e., M-H samples) in %s is incommensurate with that in %s" % (f__logL_MH,f__pdf_MH))
	sys.exit(-1)
if step_MH.shape[0] != nSamples_MH:
	print("\nERROR - Number of rows (i.e., M-H samples) in %s is incommensurate with that in %s" % (f__step_MH,f__pdf_MH))
	sys.exit(-1)

print("DONE!\n\n")


# plotting
print("Plotting...")

# log likelihood
plt.figure()
plt.plot(step_MH,logL_MH,'k',lw=linewidth)
plt.xlabel('M-H step')
plt.ylabel('log(L) / -')
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.savefig('logL.jpg')
plt.savefig('logL.eps')

# f
plt.figure()
for k in range(0,nSamples_MH):
	plt.plot(np.arange(S),f_MH[k,:],color='0.5',lw=linewidth)
plt.plot(np.arange(S),f_MAP,'r',lw=linewidth)
plt.xlabel('biased simulation i')
plt.ylabel('f$_i$ / -')
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
ax.set_yscale('log')
plt.savefig('f.jpg')
plt.savefig('f.eps')

if dim==1:
	
	# pdf
	
	# - naked
	plt.figure()
	plt.plot(binC[0],pdf_MAP,'r',lw=linewidth)
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('pdf($\psi_1$) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('pdf.jpg')
	plt.savefig('pdf.eps')
	
	# - traces
	plt.figure()
	for k in range(0,nSamples_MH):
		plt.plot(binC[0],pdf_MH[k,:],color='0.5',lw=linewidth)
	plt.plot(binC[0],pdf_MAP,'r',lw=linewidth)
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('pdf($\psi_1$) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('pdf_traces.jpg')
	plt.savefig('pdf_traces.eps')
	
	# - stdev errorbars
	pdf_MH_std = np.std(pdf_MH, axis=0)
	
	plt.figure()
	plt.plot(binC[0],pdf_MAP,'r',lw=linewidth)
	plt.plot(binC[0],pdf_MAP+pdf_MH_std,color='0.5',lw=linewidth)
	plt.plot(binC[0],pdf_MAP-pdf_MH_std,color='0.5',lw=linewidth)
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('pdf($\psi_1$) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('pdf_eb_limits.jpg')
	plt.savefig('pdf_eb_limits.eps')
	
	plt.figure()
	plt.plot(binC[0],pdf_MH_std,'b',lw=linewidth)
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('stdev(pdf($\psi_1$)) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('pdf_eb_stdev.jpg')
	plt.savefig('pdf_eb_stdev.eps')
	
	plt.figure()
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('pdf($\psi_1$) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	ax.errorbar(binC[0],pdf_MAP,yerr=pdf_MH_std,color='r',lw=linewidth)
	plt.savefig('pdf_eb_bars.jpg')
	plt.savefig('pdf_eb_bars.eps')
	
	
	# betaF
	
	# - naked
	plt.figure()
	plt.plot(binC[0],betaF_MAP,'r',lw=linewidth)
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\psi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF.jpg')
	plt.savefig('betaF.eps')
	
	# - traces
	plt.figure()
	for k in range(0,nSamples_MH):
		plt.plot(binC[0],betaF_MH[k,:],color='0.5',lw=linewidth)
	plt.plot(binC[0],betaF_MAP,'r',lw=linewidth)
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\psi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF_traces.jpg')
	plt.savefig('betaF_traces.eps')
	
	# - stdev errorbars
	betaF_MH_std = np.std(betaF_MH, axis=0)
	
	plt.figure()
	plt.plot(binC[0],betaF_MAP,'r',lw=linewidth)
	plt.plot(binC[0],betaF_MAP+betaF_MH_std,color='0.5',lw=linewidth)
	plt.plot(binC[0],betaF_MAP-betaF_MH_std,color='0.5',lw=linewidth)
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\psi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF_eb_limits.jpg')
	plt.savefig('betaF_eb_limits.eps')
	
	plt.figure()
	plt.plot(binC[0],betaF_MH_std,'b',lw=linewidth)
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('stdev($\\beta$F($\psi_1$)) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF_eb_stdev.jpg')
	plt.savefig('betaF_eb_stdev.eps')
	
	plt.figure()
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\psi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	ax.errorbar(binC[0],betaF_MAP,yerr=betaF_MH_std,color='r',lw=linewidth)
	plt.savefig('betaF_eb_bars.jpg')
	plt.savefig('betaF_eb_bars.eps')
	
	# - distributions
	plt.figure()
	
	n_betaFbins = 100
	betaFbins = np.linspace(np.amin(betaF_MH),np.amax(betaF_MH),n_betaFbins)
	betaFbin_width = betaFbins[1]-betaFbins[0]
	betaFbins_edges = np.append(betaFbins-0.5*betaFbin_width,betaFbins[-1]+0.5*betaFbin_width)
	X,Y = np.meshgrid(binC[0],betaFbins)
	Z = np.ones((M,n_betaFbins))*float('nan')
	for ii in range(0,M):
		hist, bin_edges = np.histogram(betaF_MH[:,ii],betaFbins_edges,density=True)
		Z[ii,:] = hist
	
	plt.contourf(X,Y,np.transpose(Z))
	plt.plot(binC[0],betaF_MAP,'r',lw=linewidth)
	cbar=plt.colorbar()
	plt.xlabel('$\psi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\psi_1$) / -')
	cbar.ax.set_ylabel('pdf($\\beta$F; $\psi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF_eb_dist.jpg')
	plt.savefig('betaF_eb_dist.eps')
	
elif dim==2:
	
	# pdf
	
	# - naked
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(pdf_MAP,(len(binC[0]),len(binC[1])))
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	ax.set_xlabel('$\psi_1$ / a.u.')
	ax.set_ylabel('$\psi_2$ / a.u.')
	ax.set_zlabel('pdf($\psi_1$,$\psi_2$) / (a.u.)$^{-1}$')
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('pdf.jpg')
	plt.savefig('pdf.eps')
	
	# - traces
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(pdf_MAP,(len(binC[0]),len(binC[1])))
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	for k in range(0,nSamples_MH):
		Z = np.reshape(pdf_MH[k,:],(len(binC[0]),len(binC[1])))
		ax.plot_wireframe(X,Y,np.transpose(Z),rstride=1,cstride=1,color=[0.5,0.5,0.5],linewidth=1)
	ax.set_xlabel('$\psi_1$ / a.u.')
	ax.set_ylabel('$\psi_2$ / a.u.')
	ax.set_zlabel('pdf($\psi_1$,$\psi_2$) / (a.u.)$^{-1}$')
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('pdf_traces.jpg')
	plt.savefig('pdf_traces.eps')
	
	# - stdev errorbars
	pdf_MH_std = np.std(pdf_MH, axis=0)
	
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(pdf_MAP,(len(binC[0]),len(binC[1])))
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	Z = np.reshape(pdf_MAP+pdf_MH_std,(len(binC[0]),len(binC[1])))
	ax.plot_wireframe(X,Y,np.transpose(Z),rstride=1,cstride=1,color=[0.5,0.5,0.5],linewidth=1)
	Z = np.reshape(pdf_MAP-pdf_MH_std,(len(binC[0]),len(binC[1])))
	ax.plot_wireframe(X,Y,np.transpose(Z),rstride=1,cstride=1,color=[0.5,0.5,0.5],linewidth=1)
	ax.set_xlabel('$\psi_1$ / a.u.')
	ax.set_ylabel('$\psi_2$ / a.u.')
	ax.set_zlabel('pdf($\psi_1$,$\psi_2$) / (a.u.)$^{-1}$')
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('pdf_eb_limits.jpg')
	plt.savefig('pdf_eb_limits.eps')
	
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(pdf_MH_std,(len(binC[0]),len(binC[1])))
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	ax.set_xlabel('$\psi_1$ / a.u.')
	ax.set_ylabel('$\psi_2$ / a.u.')
	ax.set_zlabel('stdev(pdf($\psi_1$,$\psi_2$)) / (a.u.)$^{-1}$')
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('pdf_eb_stdev.jpg')
	plt.savefig('pdf_eb_stdev.eps')
    
    
    # betaF
    # -> forced to set NaN to constant value -- in this case we choose 0 -- in order to plot surfaces with colored gradient
	
	# - naked
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(betaF_MAP,(len(binC[0]),len(binC[1])))
	Zmin = np.amin(Z[~np.isnan(Z)])
	Zmax = np.amax(Z[~np.isnan(Z)])
	Z[np.isnan(Z)] = 0
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	ax.set_xlabel('$\psi_1$ / a.u.')
	ax.set_ylabel('$\psi_2$ / a.u.')
	ax.set_zlabel('$\\beta$F($\psi_1$,$\psi_2$) / -')
	ax.set_zlim(Zmin, Zmax)
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('betaF.jpg')
	plt.savefig('betaF.eps')
	
	# - traces
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(betaF_MAP,(len(binC[0]),len(binC[1])))
	Zmin = np.amin(Z[~np.isnan(Z)])
	Zmax = np.amax(Z[~np.isnan(Z)])
	Z[np.isnan(Z)] = 0
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	for k in range(0,nSamples_MH):
		Z = np.reshape(betaF_MH[k,:],(len(binC[0]),len(binC[1])))
		ax.plot_wireframe(X,Y,np.transpose(Z),rstride=1,cstride=1,color=[0.5,0.5,0.5],linewidth=1)
	ax.set_xlabel('$\psi_1$ / a.u.')
	ax.set_ylabel('$\psi_2$ / a.u.')
	ax.set_zlabel('$\\beta$F($\psi_1$,$\psi_2$) / -')
	ax.set_zlim(Zmin, Zmax)
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('betaF_traces.jpg')
	plt.savefig('betaF_traces.eps')
	
	# - stdev errorbars
	betaF_MH_std = np.std(betaF_MH, axis=0)
	
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(betaF_MAP,(len(binC[0]),len(binC[1])))
	Z[np.isnan(Z)] = 0
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	Z = np.reshape(betaF_MAP+betaF_MH_std,(len(binC[0]),len(binC[1])))
	Zmax = np.amax(Z[~np.isnan(Z)])
	Z[np.isnan(Z)] = 0
	ax.plot_wireframe(X,Y,np.transpose(Z),rstride=1,cstride=1,color=[0.5,0.5,0.5],linewidth=1)
	Z = np.reshape(betaF_MAP-betaF_MH_std,(len(binC[0]),len(binC[1])))
	Zmin = np.amin(Z[~np.isnan(Z)])
	Z[np.isnan(Z)] = 0
	ax.plot_wireframe(X,Y,np.transpose(Z),rstride=1,cstride=1,color=[0.5,0.5,0.5],linewidth=1)
	ax.set_xlabel('$\psi_1$ / a.u.')
	ax.set_ylabel('$\psi_2$ / a.u.')
	ax.set_zlabel('$\\beta$F($\psi_1$,$\psi_2$) / -')
	ax.set_zlim(Zmin, Zmax)
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('betaF_eb_limits.jpg')
	plt.savefig('betaF_eb_limits.eps')
	
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(betaF_MH_std,(len(binC[0]),len(binC[1])))
	Zmin = np.amin(Z[~np.isnan(Z)])
	Zmax = np.amax(Z[~np.isnan(Z)])
	Z[np.isnan(Z)] = 0
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	ax.set_xlabel('$\psi_1$ / a.u.')
	ax.set_ylabel('$\psi_2$ / a.u.')
	ax.set_zlabel('stdev($\\beta$F($\psi_1$,$\psi_2$)) / -')
	ax.set_zlim(Zmin, Zmax)
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('betaF_eb_stdev.jpg')
	plt.savefig('betaF_eb_stdev.eps')
    
elif dim==3:
	
	# checking rectilinear bins in each dimension are all regularly spaced
	# -> mayavi only supports plotting on regularly spaced grids
	for d in range(0,2):
		if np.any(np.array(binW[d])!=binW[0][0]):
			print("\nERROR - Bins are not regularly spaced, mayavi plotting requires evenly spaced bins across all dimensions" % (d+1))
			sys.exit(-1)
	
	
	# importing mayavi module
	# \-> appears to require two attempts to successfully load
	print("\n\tLoading mayavi module required for 3D plotting...")
	try:
		from mayavi import mlab
	except:
		try:
			from mayavi import mlab
		except:
			print("\nERROR - import error on \"from mayavi import mlab\"")
			sys.exit(-1)
	print("\tDONE!\n")
	
			
	# pdf
	
	# - naked
	X,Y,Z = np.mgrid[binC[0][0]:binC[0][-1]:complex(0,len(binC[0])), binC[1][0]:binC[1][-1]:complex(0,len(binC[1])), binC[2][0]:binC[2][-1]:complex(0,len(binC[2]))]
	F = np.reshape(pdf_MAP,(len(binC[0]),len(binC[1]),len(binC[2])),order='C')
	cmin = np.amin(F[~np.isnan(F)])
	cmax = np.amax(F[~np.isnan(F)])
	
	#print("pdf:")
	#print("How many evenly spaced isosurfaces over the range [%e,%e] would you like to plot? " % (cmin,cmax))
	#n_isos = raw_input()
	n_isos = 7
	isos = np.ndarray.tolist(np.linspace(cmin,cmax,n_isos))

	mlab.figure(size=(800, 700))
	
	mlab.contour3d(X, Y, Z, F, contours=isos, colormap='jet', transparent=True, opacity=0.3)
	
	mlab.axes(extent=[binC[0][0]-0.5*binW[0][0], binC[0][-1]+0.5*binW[0][-1], binC[1][0]-0.5*binW[1][0], binC[1][-1]+0.5*binW[1][-1], binC[2][0]-0.5*binW[2][0], binC[2][-1]+0.5*binW[2][-1]])
	mlab.axes(ranges=[binC[0][0]-0.5*binW[0][0], binC[0][-1]+0.5*binW[0][-1], binC[1][0]-0.5*binW[1][0], binC[1][-1]+0.5*binW[1][-1], binC[2][0]-0.5*binW[2][0], binC[2][-1]+0.5*binW[2][-1]])
	mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, xlabel='phi_1 / a.u.', ylabel='phi_2 / a.u.', zlabel='phi_3 / a.u.', nb_labels=10)
	mlab.colorbar(title='pdf / 1/a.u.', orientation='vertical', nb_labels=10)
	mlab.view(azimuth=0, elevation=0, distance='auto', focalpoint='auto')
	
	mlab.savefig('pdf.jpg')
	mlab.close()
	
	# - stdev
	pdf_MH_std = np.std(pdf_MH, axis=0)
	
	X,Y,Z = np.mgrid[binC[0][0]:binC[0][-1]:complex(0,len(binC[0])), binC[1][0]:binC[1][-1]:complex(0,len(binC[1])), binC[2][0]:binC[2][-1]:complex(0,len(binC[2]))]
	F = np.reshape(pdf_MH_std,(len(binC[0]),len(binC[1]),len(binC[2])),order='C')
	cmin = np.amin(F[~np.isnan(F)])
	cmax = np.amax(F[~np.isnan(F)])
	
	#print("std(pdf):")
	#print("How many evenly spaced isosurfaces over the range [%e,%e] would you like to plot? " % (cmin,cmax))
	#n_isos = raw_input()
	n_isos = 7
	isos = np.ndarray.tolist(np.linspace(cmin,cmax,n_isos))

	mlab.figure(size=(800, 700))
	
	mlab.contour3d(X, Y, Z, F, contours=isos, colormap='jet', transparent=True, opacity=0.3)
	
	mlab.axes(extent=[binC[0][0]-0.5*binW[0][0], binC[0][-1]+0.5*binW[0][-1], binC[1][0]-0.5*binW[1][0], binC[1][-1]+0.5*binW[1][-1], binC[2][0]-0.5*binW[2][0], binC[2][-1]+0.5*binW[2][-1]])
	mlab.axes(ranges=[binC[0][0]-0.5*binW[0][0], binC[0][-1]+0.5*binW[0][-1], binC[1][0]-0.5*binW[1][0], binC[1][-1]+0.5*binW[1][-1], binC[2][0]-0.5*binW[2][0], binC[2][-1]+0.5*binW[2][-1]])
	mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, xlabel='phi_1 / a.u.', ylabel='phi_2 / a.u.', zlabel='phi_3 / a.u.', nb_labels=10)
	mlab.colorbar(title='std(pdf) / 1/a.u.', orientation='vertical', nb_labels=10)
	mlab.view(azimuth=0, elevation=0, distance='auto', focalpoint='auto')
	
	mlab.savefig('pdf_eb_stdev.jpg')
	mlab.close()
	
	
	# betaF
	
	# - naked
	X,Y,Z = np.mgrid[binC[0][0]:binC[0][-1]:complex(0,len(binC[0])), binC[1][0]:binC[1][-1]:complex(0,len(binC[1])), binC[2][0]:binC[2][-1]:complex(0,len(binC[2]))]
	F = np.reshape(betaF_MAP,(len(binC[0]),len(binC[1]),len(binC[2])),order='C')
	cmin = np.amin(F[~np.isnan(F)])
	cmax = np.amax(F[~np.isnan(F)])
	
	#print("betaF:")
	#print("How many evenly spaced isosurfaces over the range [%e,%e] would you like to plot? " % (cmin,cmax))
	#n_isos = raw_input()
	n_isos = 7
	isos = np.ndarray.tolist(np.linspace(cmin,cmax,n_isos))

	mlab.figure(size=(800, 700))
	
	mlab.contour3d(X, Y, Z, F, contours=isos, colormap='jet', transparent=True, opacity=0.3)
	
	mlab.axes(extent=[binC[0][0]-0.5*binW[0][0], binC[0][-1]+0.5*binW[0][-1], binC[1][0]-0.5*binW[1][0], binC[1][-1]+0.5*binW[1][-1], binC[2][0]-0.5*binW[2][0], binC[2][-1]+0.5*binW[2][-1]])
	mlab.axes(ranges=[binC[0][0]-0.5*binW[0][0], binC[0][-1]+0.5*binW[0][-1], binC[1][0]-0.5*binW[1][0], binC[1][-1]+0.5*binW[1][-1], binC[2][0]-0.5*binW[2][0], binC[2][-1]+0.5*binW[2][-1]])
	mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, xlabel='phi_1 / a.u.', ylabel='phi_2 / a.u.', zlabel='phi_3 / a.u.', nb_labels=10)
	mlab.colorbar(title='pdf / 1/a.u.', orientation='vertical', nb_labels=10)
	mlab.view(azimuth=0, elevation=0, distance='auto', focalpoint='auto')
	
	mlab.savefig('betaF.jpg')
	mlab.close()
	
	# - stdev
	betaF_MH_std = np.std(betaF_MH, axis=0)
	
	X,Y,Z = np.mgrid[binC[0][0]:binC[0][-1]:complex(0,len(binC[0])), binC[1][0]:binC[1][-1]:complex(0,len(binC[1])), binC[2][0]:binC[2][-1]:complex(0,len(binC[2]))]
	F = np.reshape(betaF_MH_std,(len(binC[0]),len(binC[1]),len(binC[2])),order='C')
	cmin = np.amin(F[~np.isnan(F)])
	cmax = np.amax(F[~np.isnan(F)])
	
	#print("std(betaF):")
	#print("How many evenly spaced isosurfaces over the range [%e,%e] would you like to plot? " % (cmin,cmax))
	#n_isos = raw_input()
	n_isos = 7
	isos = np.ndarray.tolist(np.linspace(cmin,cmax,n_isos))

	mlab.figure(size=(800, 700))
	
	mlab.contour3d(X, Y, Z, F, contours=isos, colormap='jet', transparent=True, opacity=0.3)
	
	mlab.axes(extent=[binC[0][0]-0.5*binW[0][0], binC[0][-1]+0.5*binW[0][-1], binC[1][0]-0.5*binW[1][0], binC[1][-1]+0.5*binW[1][-1], binC[2][0]-0.5*binW[2][0], binC[2][-1]+0.5*binW[2][-1]])
	mlab.axes(ranges=[binC[0][0]-0.5*binW[0][0], binC[0][-1]+0.5*binW[0][-1], binC[1][0]-0.5*binW[1][0], binC[1][-1]+0.5*binW[1][-1], binC[2][0]-0.5*binW[2][0], binC[2][-1]+0.5*binW[2][-1]])
	mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, xlabel='phi_1 / a.u.', ylabel='phi_2 / a.u.', zlabel='phi_3 / a.u.', nb_labels=10)
	mlab.colorbar(title='std(pdf) / 1/a.u.', orientation='vertical', nb_labels=10)
	mlab.view(azimuth=0, elevation=0, distance='auto', focalpoint='auto')
	
	mlab.savefig('betaF_eb_stdev.jpg')
	mlab.close()

else:

	print("\nERROR - Dimensionality of data dim = %d; plotting only available for {1,2,3}-dimensional space" % (dim))
	sys.exit(-1)

print("DONE!\n\n")
