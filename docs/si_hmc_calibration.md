# HMC calibration and convergence assessment

We assessed convergence of the No-U-Turn Sampler (NUTS) used to propagate GP
hyperparameter uncertainty by varying both the adaptation (warmup) period and
the number of retained posterior samples. Four data regimes were considered:
an easy case using 25 umbrella windows and the full post-equilibration
trajectory, a medium case using 13 windows and one half of each trajectory, a
hard case using 7 windows and one quarter of each trajectory, and a
deliberately extreme super-hard case using only 3 windows and one tenth of each
trajectory. In every case, four independent chains were run for each
combination of 250, 500, or 1000 warmup steps and 250, 500, or 1000 retained
samples per chain. This produced 36 calibration runs and covered conditions
ranging from the data-rich calculation to a limit with insufficient spatial
and trajectory coverage. All runs used the stationary kernel and the
leave-one-out objective employed in the corresponding uncertainty analysis.

Convergence was evaluated using the potential scale reduction factor,
\(\hat R\), the effective sample size (ESS), the number of divergent
transitions, and visual comparison of the four chain traces and marginal
histograms for each inferred hyperparameter. We additionally compared
posterior-derived quantities across run lengths, including the average
predictive standard deviation and the inferred free-energy barrier. This last
comparison tests whether residual Monte Carlo variation materially affects the
reported physical observables.

The chains were generally well mixed over the full calibration grid. No
divergent transitions occurred in any of the 36 runs. For the easy and medium
data regimes, the largest \(\hat R\) observed over all hyperparameters and run
lengths was 1.0062 and 1.0033, respectively. In the hard regime, eight of the
nine runs had \(\hat R < 1.01\). The exception was the shortest calculation,
with 250 warmup steps and 250 retained samples per chain, for which the maximum
\(\hat R\) was 1.0196 and the minimum ESS was 232.

The super-hard case provided a more stringent test because its hyperparameters
were only weakly identified by the data. Seven of its nine runs had
\(\hat R < 1.01\). The two marginal cases used only 250 retained samples, with
maximum \(\hat R\) values of 1.0107 and 1.0110 after 500 and 1000 warmup steps,
respectively. In contrast, all super-hard runs with 500 retained samples had
\(\hat R < 1.005\), and all runs with 1000 retained samples had
\(\hat R < 1.002\), including the calculation with only 250 warmup steps. These
results indicate that 250 retained samples can be inadequate for weakly
identified posteriors, whereas increasing the retained sample count was more
beneficial than extending warmup beyond 250 steps. Based on these results, we
selected 500 warmup steps followed by 1000 retained samples per chain for the
production calculations. The 1000-warmup/1000-sample calculations provide a
more conservative sensitivity check but did not materially improve the
diagnostics or posterior observables.

The production setting of 500 warmup steps followed by 1000 retained samples
per chain was converged in all four regimes (Table S1). The maximum \(\hat R\)
was 1.0024, the minimum ESS was 1479, and no divergences were observed. Trace
plots at this setting showed stationary fluctuations without persistent
between-chain offsets, while the marginal histograms from the four chains
overlapped closely. Even the lowest ESS, obtained for the super-hard case,
remained comfortably above 1000 effective samples.

**Table S1. NUTS diagnostics and posterior observables using 500 warmup steps
and 1000 retained samples per chain.**

| Data regime | Windows | Trajectory fraction | Maximum \(\hat R\) | Minimum ESS | Divergences | Mean predictive SD (kJ mol\(^{-1}\)) | Barrier mean (kJ mol\(^{-1}\)) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 25 | 1.00 | 1.0008 | 2823 | 0 | 4.308 | 48.645 |
| Medium | 13 | 0.50 | 1.0012 | 2393 | 0 | 5.738 | 50.586 |
| Hard | 7 | 0.25 | 1.0023 | 1554 | 0 | 4.687 | 36.863 |
| Super hard | 3 | 0.10 | 1.0018 | 1479 | 0 | 9.978 | 30.457 |

Posterior predictions were also insensitive to the precise warmup and sampling
length once the chains passed the convergence diagnostics. Across all nine
warmup/sample combinations, the coefficient of variation of the mean
predictive standard deviation was 1.66%, 1.31%, 1.84%, and 2.93% for the easy,
medium, hard, and super-hard regimes, respectively. The corresponding barrier
estimates spanned 48.569--48.688, 50.402--50.723, 36.841--36.908, and
30.457--30.942 kJ mol\(^{-1}\). The largest within-regime range in the barrier
mean was therefore 0.486 kJ mol\(^{-1}\), observed in the super-hard case, and
the production-run estimates lay within these ranges.

Importantly, convergence of the sampler did not imply that the available data
were sufficient to reconstruct the PMF. At the recommended
500-warmup/1000-sample setting, the super-hard chains were well mixed, but the
mean predictive standard deviation increased to 9.98 kJ mol\(^{-1}\), compared
with 4.31--5.74 kJ mol\(^{-1}\) in the other regimes. The super-hard
reconstruction also had RMSE values of 21.06 and 20.84 kJ mol\(^{-1}\) relative
to the WHAM and umbrella-integration references, respectively. The calibration
therefore separates two distinct failure modes: inadequate HMC run length is
detectable through \(\hat R\) and ESS, whereas inadequate simulation data can
remain even when the posterior sampler itself has converged.

Together, the absence of divergences, near-unity \(\hat R\), large effective
sample sizes, overlapping chain distributions, and stability of the
posterior-derived observables support the conclusion that four chains with
500 warmup steps and 1000 retained samples per chain are sufficient to sample
the stationary-kernel hyperposterior over all data regimes tested, including
the deliberately extreme three-window case. The calibration also shows that
shorter calculations can be adequate, but that only 250 retained samples
should be avoided for strongly data-limited calculations. These diagnostics
validate the HMC run length; they do not validate PMF accuracy when the
underlying umbrella-sampling data are insufficient.

**Suggested Figure S1 caption.** NUTS calibration across data availability and
run length. Maximum \(\hat R\) and minimum effective sample size are shown for
four independent chains using 250, 500, or 1000 warmup steps and 250, 500, or
1000 retained samples per chain. No divergent transitions were observed. Runs
with 500 or 1000 retained samples had \(\hat R < 1.01\) in every data regime;
the three runs exceeding this threshold used only 250 retained samples.

**Suggested Figure S2 caption.** Representative NUTS traces and marginal
posterior histograms for the GP hyperparameters using 500 warmup steps and
1000 retained samples per chain. The four chains fluctuate around common
stationary distributions and yield closely overlapping marginal histograms in
the easy, medium, hard, and super-hard data regimes.
