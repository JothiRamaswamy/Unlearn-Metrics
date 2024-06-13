# Folder Contents

`seedx_FTy_GAz`: folder containing results for fine tuning methods on y epochs (y is one of 10, 15, 20) and gradient ascent methods on z epochs (z is one of 5, 7, 10), both with a seed of x (x is one of 1, 2, 3) for randomly choosing the forget set. FT and GA models grouped together for convenience of running models together.

## Subfolder Contents

`FT_metrics.json`: results for fine tuned method, given the specified epochs + seed in the subfolder.

`GA_metrics.json`: results for gradient method, given the specified epochs + seed in the subfolder.

`all_metrics.json`: all subfolder metrics combined.

`retrain_metrics.json`: results for retrain method, given the specified seed in the subfolder.
