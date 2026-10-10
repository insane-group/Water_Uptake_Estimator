# Archived manuscript results

These files preserve the numerical records used for the revised manuscript and supplement. They were copied unchanged from the local revision analyses. Names retain comment identifiers for cross-referencing.

Generated files go to `../results/` by default, not this directory. Source hashes inside archived records belong to historical analysis versions; new reports record current source hashes.

The NPZ archive has arrays named `rsd_<level>_<property>`, each of shape `(512, 23)`. Levels are `0.005`, `0.01`, `0.02`; properties are `mass`, `thick`, `dens`, `cg`, `proxy`, `youngs`. Quantiles of interpolated draws give the five-record test envelopes. Interpolating grid quantiles is not equivalent.

These results do not establish independent-panel testing, calibrated uncertainty, or complete regeneration of physical wave anchors. See the main README for data access and physical assumptions.
