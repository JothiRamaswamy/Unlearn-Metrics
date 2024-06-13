<div align="center">

# EVALUATING VARIOUS METRICS FOR MEASURING THE EFFICACY OF MACHINE UNLEARNING ALGORITHMS

## Files changed/added in forked repository

### Added Folders

`results`: contains resulting metrics of each of the models and in each unique seed (used for selecting forget set that model leverages for unlearning). More details in README.md file inside folder.

`final_metrics`: contains final metrics of each of the models, averaged across all seeds (used for selecting forget set that model leverages for unlearning). More details in README.md file inside folder

### Added Files

`load_model.py`: contains code for saving (after untarring) + loading the model after unlearning (in the FASRC cluster), obtaining model predictions + prediction logits, obtaining the forget set used for unlearning

`metrics.py`: code for generating metric results given model weights + predictions (and logits). Metrics created in this file: wasserstein distance on forget/test set predictions in output space, KL divergence of model weights in weight space, L_2 distance of model weights in weight space.

`metric_analysis.py`: contains primary code for generating plots of metrics obtained from models after unlearning.
