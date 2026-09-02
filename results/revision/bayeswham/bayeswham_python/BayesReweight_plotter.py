"""

% Copyright:	Andrew L. Ferguson, UIUC 
% Last updated:	2 Jan 2016

% SYNOPSIS
%
% code to perform plotting of: 	(i)   {1,2,3}-dimensional maximum a posteriori (MAP) free energy surfaces (FES) computed by Bayesian inference of biased umbrella sampling trajectories reweighted into projection variables xi 
%								(ii)  uncertainty estimates in MAP FES estimated using Metropolis-Hastings samples from the Bayes posterior of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i for i=1..S biased simulations as estimates of distribution of free energy shifts between the biased simulations 

% INPUTS
%
% f__hist_binCenters  		- [str] [dim x M_k float] path to text file containing dim rows specifying M_k k=1..dim centers of the rectilinear histogram bins in each dimension used to construct reweighted dim-dimensional histograms constituting f__pdf_MAP/MH and f__betaF_MAP/MH 
% f__hist_binWidths   		- [str] [dim x M_k float] path to text file containing dim rows specifying M_k k=1..dim widths of the rectilinear histogram bins in each dimension used to construct reweighted dim-dimensional histograms constituting f__pdf_MAP/MH and f__betaF_MAP/MH 
% f__pdf_MAP          		- [str] [1 x M float] path to text file containing reweighted MAP estimate of unbiased probability density function pdf_l over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) 
% f__betaF_MAP        		- [str] [1 x M float] path to text file containing reweighted MAP estimate of unbiased free energy surface beta*F_l = -ln(p(xi)/binVolume) + const. over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) 
% f__pdf_MH           		- [str] [nSamples_MH x M float] path to text file containing reweighted nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased probability density function pdf_l over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) 
% f__betaF_MH         		- [str] [nSamples_MH x M float] path to text file containing reweighted nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased free energy surface beta*F_l = -ln(p(xi)/binVolume) + const. over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) 

% OUTPUTS
%
% 1-dimension:
%
% pdf_PROJ.jpg/eps				- reweighted MAP estimate of probability density function as a function of xi 
% pdf_traces_PROJ.jpg/eps		- reweighted MAP estimate of probability density function as a function of xi overlaid with traces of reweighted MH samples 
% pdf_eb_limits_PROJ.jpg/eps	- reweighted MAP estimate of probability density function as a function of xi overlaid with the reweighted MAP estimates plus and minus the standard deviation of the reweighted ensemble of MH samples 
% pdf_eb_stdev_PROJ.jpg/eps		- standard deviation of the reweighted probability density function of the ensemble of MH samples as a function of xi 
% pdf_eb_bars_PROJ.jpg/eps		- reweighted MAP estimate of probability density function as a function of xi with error bars denoting the standard deviation of the reweighted ensemble of MH samples 
% betaF_PROJ.jpg/eps			- reweighted MAP estimate of beta*F = -ln(p(xi)/binVolume) + const. as a function of xi 
% betaF_traces_PROJ.jpg/eps		- reweighted MAP estimate of beta*F = -ln(p(xi)/binVolume) + const. as a function of xi overlaid with traces of reweighted MH samples 
% betaF_eb_limits_PROJ.jpg/eps	- reweighted MAP estimate of beta*F = -ln(p(xi)/binVolume) + const. as a function of xi overlaid with the reweighted MAP estimates plus and minus the standard deviation of the reweighted ensemble of MH samples 
% betaF_eb_stdev_PROJ.jpg/eps	- standard deviation of the reweighted beta*F = -ln(p(xi)/binVolume) + const. of the ensemble of MH samples as a function of xi 
% betaF_eb_bars_PROJ.jpg/eps	- reweighted MAP estimate of beta*F = -ln(p(xi)/binVolume) + const. as a function of xi with error bars denoting the standard deviation of the reweighted ensemble of MH samples 
% betaF_eb_dist_PROJ.jpg/eps	- reweighted MAP estimate of beta*F = -ln(p(xi)/binVolume) + const. as a function of xi overlaid with the probability density function of the reweighted MH ensemble at each MAP data point 
%
% 2-dimensions:
%
% pdf_PROJ.jpg/eps				- reweighted MAP estimate of probability density function as a function of xi 
% pdf_traces_PROJ.jpg/eps		- reweighted MAP estimate of probability density function as a function of xi overlaid with traces of reweighted MH samples 
% pdf_eb_limits_PROJ.jpg/eps	- reweighted MAP estimate of probability density function as a function of xi overlaid with the reweighted MAP estimates plus and minus the standard deviation of the reweighted ensemble of MH samples 
% pdf_eb_stdev_PROJ.jpg/eps		- standard deviation of the reweighted probability density function of the ensemble of MH samples as a function of xi 
% betaF_PROJ.jpg/eps			- reweighted MAP estimate of beta*F = -ln(p(xi)/binVolume) + const. as a function of xi 
% betaF_traces_PROJ.jpg/eps		- reweighted MAP estimate of beta*F = -ln(p(xi)/binVolume) + const. as a function of xi overlaid with traces of reweighted MH samples 
% betaF_eb_limits_PROJ.jpg/eps	- reweighted MAP estimate of beta*F = -ln(p(xi)/binVolume) + const. as a function of xi overlaid with the reweighted MAP estimates plus and minus the standard deviation of the reweighted ensemble of MH samples 
% betaF_eb_stdev_PROJ.jpg/eps	- standard deviation of the reweighted beta*F = -ln(p(xi)/binVolume) + const. of the ensemble of MH samples as a function of xi 
%
% 3-dimensions:
%
% pdf_PROJ.jpg					- reweighted MAP estimate of probability density function as a function of xi 
% pdf_eb_stdev_PROJ.jpg			- standard deviation of the reweighted probability density function of the ensemble of MH samples as a function of xi 
% betaF_PROJ.jpg				- reweighted MAP estimate of beta*F = -ln(p(xi)/binVolume) + const. as a function of xi 
% betaF_eb_stdev_PROJ.jpg		- standard deviation of the reweighted beta*F = -ln(p(xi)/binVolume) + const. of the ensemble of MH samples as a function of xi 

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
	print "USAGE: %s f__hist_binCenters f__hist_binWidths f__pdf_MAP f__betaF_MAP f__pdf_MH f__betaF_MH" % sys.argv[0]
	print "       f__hist_binCenters  - [str] [dim x M_k float] path to text file containing dim rows specifying M_k k=1..dim centers of the rectilinear histogram bins in each dimension used to construct reweighted dim-dimensional histograms constituting f__pdf_MAP/MH and f__betaF_MAP/MH "
	print "       f__hist_binWidths   - [str] [dim x M_k float] path to text file containing dim rows specifying M_k k=1..dim widths of the rectilinear histogram bins in each dimension used to construct reweighted dim-dimensional histograms constituting f__pdf_MAP/MH and f__betaF_MAP/MH "
	print "       f__pdf_MAP          - [str] [1 x M float] path to text file containing reweighted MAP estimate of unbiased probability density function pdf_l over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) "
	print "       f__betaF_MAP        - [str] [1 x M float] path to text file containing reweighted MAP estimate of unbiased free energy surface beta*F_l = -ln(p(xi)/binVolume) + const. over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) "
	print "       f__pdf_MH           - [str] [nSamples_MH x M float] path to text file containing reweighted nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased probability density function pdf_l over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) "
	print "       f__betaF_MH         - [str] [nSamples_MH x M float] path to text file containing reweighted nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased free energy surface beta*F_l = -ln(p(xi)/binVolume) + const. over l=1..M bins of dim-dimensional rectilinear histogram in row major order (last index changes fastest) "
	print "       OR"
	print "USAGE: %s" % sys.argv[0]
	print "       \-> to accept default arguments:"
	print "       f__hist_binCenters  = hist_binCenters_PROJ.txt"
	print "       f__hist_binWidths   = hist_binWidths_PROJ.txt"
	print "       f__pdf_MAP          = pdf_PROJ_MAP.txt"
	print "       f__betaF_MAP        = betaF_PROJ_MAP.txt"
	print "       f__pdf_MH           = pdf_PROJ_MH.txt"
	print "       f__betaF_MH         = betaF_PROJ_MH.txt"
	
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
	f__hist_binCenters = 'hist_binCenters_PROJ.txt'
	f__hist_binWidths = 'hist_binWidths_PROJ.txt'
	f__pdf_MAP = 'pdf_PROJ_MAP.txt'
	f__betaF_MAP = 'betaF_PROJ_MAP.txt'
	f__pdf_MH = 'pdf_PROJ_MH.txt'
	f__betaF_MH = 'betaF_PROJ_MH.txt'
elif len(sys.argv) == 7:
	f__hist_binCenters = str(sys.argv[1])
	f__hist_binWidths = str(sys.argv[2])
	f__pdf_MAP = str(sys.argv[3])
	f__betaF_MAP = str(sys.argv[4])
	f__pdf_MH = str(sys.argv[5])
	f__betaF_MH = str(sys.argv[6])
else:
	_usage()
	sys.exit(-1)

# - printing args to screen
print("")
print("f__hist_binCenters = %s" % (f__hist_binCenters))
print("f__hist_binWidths = %s" % (f__hist_binWidths))
print("f__pdf_MAP = %s" % (f__pdf_MAP))
print("f__betaF_MAP = %s" % (f__betaF_MAP))
print("f__pdf_MH = %s" % (f__pdf_MH))
print("f__betaF_MH = %s" % (f__betaF_MH))
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

print("DONE!\n\n")


# plotting
print("Plotting...")

if dim==1:
	
	# pdf
	
	# - naked
	plt.figure()
	plt.plot(binC[0],pdf_MAP,'r',lw=linewidth)
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('pdf($\\xi_1$) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('pdf_PROJ.jpg')
	plt.savefig('pdf_PROJ.eps')
	
	# - traces
	plt.figure()
	for k in range(0,nSamples_MH):
		plt.plot(binC[0],pdf_MH[k,:],color='0.5',lw=linewidth)
	plt.plot(binC[0],pdf_MAP,'r',lw=linewidth)
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('pdf($\\xi_1$) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('pdf_traces_PROJ.jpg')
	plt.savefig('pdf_traces_PROJ.eps')
	
	# - stdev errorbars
	pdf_MH_std = np.std(pdf_MH, axis=0)
	
	plt.figure()
	plt.plot(binC[0],pdf_MAP,'r',lw=linewidth)
	plt.plot(binC[0],pdf_MAP+pdf_MH_std,color='0.5',lw=linewidth)
	plt.plot(binC[0],pdf_MAP-pdf_MH_std,color='0.5',lw=linewidth)
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('pdf($\\xi_1$) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('pdf_eb_limits_PROJ.jpg')
	plt.savefig('pdf_eb_limits_PROJ.eps')
	
	plt.figure()
	plt.plot(binC[0],pdf_MH_std,'b',lw=linewidth)
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('stdev(pdf($\\xi_1$)) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('pdf_eb_stdev_PROJ.jpg')
	plt.savefig('pdf_eb_stdev_PROJ.eps')
	
	plt.figure()
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('pdf($\\xi_1$) / (a.u.)$^{-1}$')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	ax.errorbar(binC[0],pdf_MAP,yerr=pdf_MH_std,color='r',lw=linewidth)
	plt.savefig('pdf_eb_bars_PROJ.jpg')
	plt.savefig('pdf_eb_bars_PROJ.eps')
	
	
	# betaF
	
	# - naked
	plt.figure()
	plt.plot(binC[0],betaF_MAP,'r',lw=linewidth)
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\\xi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF_PROJ.jpg')
	plt.savefig('betaF_PROJ.eps')
	
	# - traces
	plt.figure()
	for k in range(0,nSamples_MH):
		plt.plot(binC[0],betaF_MH[k,:],color='0.5',lw=linewidth)
	plt.plot(binC[0],betaF_MAP,'r',lw=linewidth)
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\\xi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF_traces_PROJ.jpg')
	plt.savefig('betaF_traces_PROJ.eps')
	
	# - stdev errorbars
	betaF_MH_std = np.std(betaF_MH, axis=0)
	
	plt.figure()
	plt.plot(binC[0],betaF_MAP,'r',lw=linewidth)
	plt.plot(binC[0],betaF_MAP+betaF_MH_std,color='0.5',lw=linewidth)
	plt.plot(binC[0],betaF_MAP-betaF_MH_std,color='0.5',lw=linewidth)
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\\xi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF_eb_limits_PROJ.jpg')
	plt.savefig('betaF_eb_limits_PROJ.eps')
	
	plt.figure()
	plt.plot(binC[0],betaF_MH_std,'b',lw=linewidth)
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('stdev($\\beta$F($\\xi_1$)) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF_eb_stdev_PROJ.jpg')
	plt.savefig('betaF_eb_stdev_PROJ.eps')
	
	plt.figure()
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\\xi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	ax.errorbar(binC[0],betaF_MAP,yerr=betaF_MH_std,color='r',lw=linewidth)
	plt.savefig('betaF_eb_bars_PROJ.jpg')
	plt.savefig('betaF_eb_bars_PROJ.eps')
	
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
	plt.xlabel('$\\xi_1$ / a.u.')
	plt.ylabel('$\\beta$F($\\xi_1$) / -')
	cbar.ax.set_ylabel('pdf($\\beta$F; $\\xi_1$) / -')
	ax = plt.gca()
	ax.ticklabel_format(useOffset=False)
	plt.savefig('betaF_eb_dist_PROJ.jpg')
	plt.savefig('betaF_eb_dist_PROJ.eps')
	
elif dim==2:
	
	# pdf
	
	# - naked
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(pdf_MAP,(len(binC[0]),len(binC[1])))
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	ax.set_xlabel('$\\xi_1$ / a.u.')
	ax.set_ylabel('$\\xi_2$ / a.u.')
	ax.set_zlabel('pdf($\\xi_1$,$\\xi_2$) / (a.u.)$^{-1}$')
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('pdf_PROJ.jpg')
	plt.savefig('pdf_PROJ.eps')
	
	# - traces
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(pdf_MAP,(len(binC[0]),len(binC[1])))
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	for k in range(0,nSamples_MH):
		Z = np.reshape(pdf_MH[k,:],(len(binC[0]),len(binC[1])))
		ax.plot_wireframe(X,Y,np.transpose(Z),rstride=1,cstride=1,color=[0.5,0.5,0.5],linewidth=1)
	ax.set_xlabel('$\\xi_1$ / a.u.')
	ax.set_ylabel('$\\xi_2$ / a.u.')
	ax.set_zlabel('pdf($\\xi_1$,$\\xi_2$) / (a.u.)$^{-1}$')
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('pdf_traces_PROJ.jpg')
	plt.savefig('pdf_traces_PROJ.eps')
	
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
	ax.set_xlabel('$\\xi_1$ / a.u.')
	ax.set_ylabel('$\\xi_2$ / a.u.')
	ax.set_zlabel('pdf($\\xi_1$,$\\xi_2$) / (a.u.)$^{-1}$')
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('pdf_eb_limits_PROJ.jpg')
	plt.savefig('pdf_eb_limits_PROJ.eps')
	
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(pdf_MH_std,(len(binC[0]),len(binC[1])))
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	ax.set_xlabel('$\\xi_1$ / a.u.')
	ax.set_ylabel('$\\xi_2$ / a.u.')
	ax.set_zlabel('stdev(pdf($\\xi_1$,$\\xi_2$)) / (a.u.)$^{-1}$')
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('pdf_eb_stdev_PROJ.jpg')
	plt.savefig('pdf_eb_stdev_PROJ.eps')
    
    
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
	ax.set_xlabel('$\\xi_1$ / a.u.')
	ax.set_ylabel('$\\xi_2$ / a.u.')
	ax.set_zlabel('$\\beta$F($\\xi_1$,$\\xi_2$) / -')
	ax.set_zlim(Zmin, Zmax)
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('betaF_PROJ.jpg')
	plt.savefig('betaF_PROJ.eps')
	
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
	ax.set_xlabel('$\\xi_1$ / a.u.')
	ax.set_ylabel('$\\xi_2$ / a.u.')
	ax.set_zlabel('$\\beta$F($\\xi_1$,$\\xi_2$) / -')
	ax.set_zlim(Zmin, Zmax)
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('betaF_traces_PROJ.jpg')
	plt.savefig('betaF_traces_PROJ.eps')
	
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
	ax.set_xlabel('$\\xi_1$ / a.u.')
	ax.set_ylabel('$\\xi_2$ / a.u.')
	ax.set_zlabel('$\\beta$F($\\xi_1$,$\\xi_2$) / -')
	ax.set_zlim(Zmin, Zmax)
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('betaF_eb_limits_PROJ.jpg')
	plt.savefig('betaF_eb_limits_PROJ.eps')
	
	plt.figure()
	ax = plt.gca(projection='3d')
	X,Y = np.meshgrid(binC[0],binC[1])
	Z = np.reshape(betaF_MH_std,(len(binC[0]),len(binC[1])))
	Zmin = np.amin(Z[~np.isnan(Z)])
	Zmax = np.amax(Z[~np.isnan(Z)])
	Z[np.isnan(Z)] = 0
	ax.plot_surface(X,Y,np.transpose(Z),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=1)
	ax.set_xlabel('$\\xi_1$ / a.u.')
	ax.set_ylabel('$\\xi_2$ / a.u.')
	ax.set_zlabel('stdev($\\beta$F($\\xi_1$,$\\xi_2$)) / -')
	ax.set_zlim(Zmin, Zmax)
	ax.ticklabel_format(useOffset=False, style='sci', scilimits=(-3,3))
	plt.savefig('betaF_eb_stdev_PROJ.jpg')
	plt.savefig('betaF_eb_stdev_PROJ.eps')
    
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
	
	mlab.savefig('pdf_PROJ.jpg')
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
	
	mlab.savefig('pdf_eb_stdev_PROJ.jpg')
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
	
	mlab.savefig('betaF_PROJ.jpg')
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
	
	mlab.savefig('betaF_eb_stdev_PROJ.jpg')
	mlab.close()

else:

	print("\nERROR - Dimensionality of data dim = %d; plotting only available for {1,2,3}-dimensional space" % (dim))
	sys.exit(-1)

print("DONE!\n\n")
