# Length-scale prior sensitivity

We assessed sensitivity of the GP reconstruction to the prior on the kernel
length scale, \(\ell\), in a deliberately data-limited setting containing seven
umbrella windows and one quarter of each post-equilibration trajectory. Four
priors were compared: a bounded prior uniform in \(\log \ell\), the default
\(\log \ell \sim \mathcal{N}(\log 4, 1)\) prior, and two priors centered at
\(\ell=0.5\) nm with standard deviations of 0.5 and 0.1 in log space. Priors on
the kernel amplitude and the function and derivative noise parameters were
unchanged. Each hyperposterior was sampled using four independent NUTS chains
with 500 warmup steps and 1000 retained samples per chain.

The inferred PMF was insensitive to the flat, default, and moderately
informative priors (Table S2). Their posterior median length scales ranged only
from 0.0604 to 0.0609 nm, and their 90% credible intervals strongly overlapped.
Their hyperposterior-propagated PMFs were visually indistinguishable, with mean
predictive standard deviations of 4.71--4.92 kJ mol\(^{-1}\) and RMSE values
of 17.88--17.94 kJ mol\(^{-1}\) relative to the umbrella-integration
reference. The broad default prior therefore does not appear to determine the
result, even with substantially reduced data.

The very narrow prior centered at 0.5 nm illustrates the stronger
regularization available in low-data regimes. It shifted the posterior median
length scale to 0.244 nm and produced a substantially smoother PMF. This
regularization was too strong for the present choice of prior center: the RMSE
relative to umbrella integration increased to 29.98 kJ mol\(^{-1}\), and the
mean predictive standard deviation increased to 11.88 kJ mol\(^{-1}\). Thus,
an informative prior can stabilize the reconstruction toward a chosen
smoothness scale when data are limited, but that scale must be physically
justified rather than imposed solely to suppress short-length-scale structure.

**Table S2. Sensitivity of the data-limited reconstruction to the length-scale
prior. Credible intervals are the 5th and 95th hyperposterior percentiles.**

| Length-scale prior | Median \(\ell\) (nm) | 90% interval (nm) | Mean predictive SD (kJ mol\(^{-1}\)) | RMSE vs UI (kJ mol\(^{-1}\)) | Maximum \(\hat R\) | Minimum ESS |
|---|---:|---:|---:|---:|---:|---:|
| Flat in \(\log \ell\) | 0.0604 | 0.0573--0.0639 | 4.71 | 17.94 | 1.0007 | 1117 |
| Default: \(\mathcal{N}(\log 4,1)\) | 0.0605 | 0.0575--0.0639 | 4.72 | 17.94 | 1.0009 | 1465 |
| Centered at 0.5 nm, \(\sigma_{\log}=0.5\) | 0.0609 | 0.0578--0.0644 | 4.92 | 17.88 | 1.0040 | 1178 |
| Centered at 0.5 nm, \(\sigma_{\log}=0.1\) | 0.2443 | 0.2187--0.2741 | 11.88 | 29.98 | 1.0019 | 2187 |

All four calculations were well sampled: the maximum \(\hat R\) was 1.0040,
the minimum effective sample size was 1117, and no divergent transitions were
observed. The differences produced by the very narrow prior therefore reflect
prior sensitivity rather than failure of the HMC calculation.

These results show that the reconstruction is robust to broad and moderately
informative choices of the length-scale prior. Stronger regularization remains
available for more weakly identified low-data calculations, but a narrow prior
can bias the inferred surface when its preferred scale conflicts with the
data. We therefore retain the broad default prior for the reported analysis.

This sensitivity analysis is necessarily representative rather than
exhaustive. It does not test every combination of data availability, prior
family, prior location, or prior width, and therefore cannot establish
prior-insensitivity under all possible analysis conditions. Instead, it
provides a targeted stress test in a deliberately data-limited regime, where
prior dependence should be more apparent than in data-rich calculations. The
results support the more limited conclusion that the reported reconstruction
is insensitive to the broad and moderately informative length-scale priors
tested here, whereas sufficiently concentrated priors can materially influence
the posterior when they impose a competing smoothness scale. Prior sensitivity
should therefore be reassessed if substantially different data regimes or
strongly informative hyperpriors are used.

**Suggested Figure S3 caption.** Sensitivity of the data-limited GP
reconstruction to the length-scale prior. Top: posterior mean PMFs and
hyperposterior-propagated uncertainty bands for seven umbrella windows and one
quarter of each trajectory; the block-averaged umbrella-integration estimate
is shown as a reference. Bottom: direct comparison of the length-scale priors
(dashed lines) and kernel-density estimates of the sampled hyperposteriors
(solid lines), each normalized to unit peak height to facilitate comparison of
their locations and shapes. Colors identify the prior specification. Flat,
default, and moderately informative priors yield nearly identical
hyperposteriors and PMFs. A very narrow prior centered at 0.5 nm imposes
stronger smoothing but increases uncertainty and reference error for this data
set.
