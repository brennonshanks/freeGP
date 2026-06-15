# GP hyperpriors and observation-noise model

For the stationary-kernel analyses, the free-energy surface was assigned a
squared-exponential covariance,
\[
k(x,x')=w^2\exp\left[-\frac{(x-x')^2}{2\ell^2}\right],
\]
where \(\ell\) is the correlation length and \(w\) is the marginal standard
deviation of the free-energy function. All positive hyperparameters were
sampled in natural-log space. Expressing lengths in nm and energies in
kJ mol\(^{-1}\), the default independent hyperpriors were
\[
\begin{aligned}
\log \ell &\sim \mathcal{N}(\log 4,\,1^2),\\
\log w &\sim \mathcal{N}(1,\,0.5^2),\\
\log \sigma_f &\sim \mathcal{N}(0.5,\,2^2),\\
\log \sigma_d &\sim \mathcal{N}(0.5,\,2^2).
\end{aligned}
\]
Equivalently, \(\ell\), \(w\), \(\sigma_f\), and \(\sigma_d\) have log-normal
priors with medians of 4 nm, 2.72 kJ mol\(^{-1}\), 1.65 kJ mol\(^{-1}\), and
1.65 kJ mol\(^{-1}\) nm\(^{-1}\), respectively. The parameters
\(\sigma_f\) and \(\sigma_d\) are nuisance noise scales: their variances,
\(\sigma_f^2\) and \(\sigma_d^2\), were added to the diagonal of the function
and derivative observation covariance blocks. The priors on these noise
scales were intentionally broad, while the prior on \(w\) provided mild
regularization of the function amplitude. Unless stated otherwise, these
defaults were used for both hyperparameter optimization and
hyperposterior-propagated uncertainty calculations.

The fixed-hyperparameter reference used the same stationary covariance with
\(\ell=\pi/2\) nm and \(w=4.184\sqrt{10}\) kJ mol\(^{-1}\). Although these
kernel parameters were fixed, the observation covariance was estimated from
each umbrella trajectory. After equilibration removal, the position series in
each window was approximated as an autoregressive process of order one. The
lag-one coefficient was estimated using
\[
\hat{\phi}
=
\tanh\left[
\frac{\sum_t (x_t-\bar{x})(x_{t+1}-\bar{x})}
{0.01+\sum_t (x_t-\bar{x})^2}
\right],
\]
where the constant 0.01 is the precision of a zero-centered regularizing
prior. The corresponding statistical inefficiency and effective sample size
were
\[
\tau=\frac{1+\hat{\phi}}{1-\hat{\phi}},
\qquad
N_{\mathrm{eff}}=\frac{N}{\tau}.
\]

For a histogram bin with probability \(p_i\), uncertainty in the
histogram-derived free-energy observation was represented by
\[
\operatorname{Var}(F_i)
=
\frac{1}{\beta^2N_{\mathrm{eff}}}
\left(\frac{1}{p_i}-1\right).
\]
The multinomial constraint introduces covariance between bins from the same
umbrella window,
\[
\operatorname{Cov}(F_i,F_j)
=
-\frac{1}{\beta^2N_{\mathrm{eff}}},
\qquad i\ne j.
\]
For the derivative observation obtained from an umbrella with force constant
\(k\) and positional variance \(s_x^2\), the estimated variance was
\[
\operatorname{Var}(\overline{F'})
=
\frac{k^2s_x^2}{N_{\mathrm{eff}}}.
\]
These trajectory-derived covariance terms were supplied directly to the fixed
GP. Thus, “fixed hyperparameters” refers to the kernel length and amplitude,
not to an assumption of uniform or known observation noise.

For the optimized and hyperposterior calculations, the covariance model
instead used the inferred nuisance scales \(\sigma_f^2 I\) and
\(\sigma_d^2 I\) defined above. The AR(1) estimate should be interpreted as a
computationally inexpensive approximation to the integrated autocorrelation
time. It is exact only under the assumed single-timescale AR(1) model and may
not capture more complicated or multiscale temporal correlations.
