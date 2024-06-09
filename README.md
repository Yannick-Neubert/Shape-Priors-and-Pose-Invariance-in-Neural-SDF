# Shape Priors and Pose Invariance in Neural SDF

This repository provides code for my master's thesis about learning a pose-invariant latent shape representation. 


# Usage

## Reproducing Thesis Results
To reproduce the main results presented in the thesis, simply run `eval.py` as is. This evaluates the model trained on second-order moments with respect to accuracy and orientation of trained and inferred latent shape representations.

### Further Results
The file `workspace.py` contains all controllable parameters to train, use, and evaluate a model. 

Further results, such as those obtained by training the model on third-order moments can be reproduced by setting the workspace parameters `order = 3` or `pose_inference = True` for evaluating pose and code inference and then rerunning `eval.py`. 

`eval.py` also presents qualitative evaluation methods, which can be run by calling the different methods in `eval.py`.

## Running Experiments
The file `workspace.py` can be adapted to change moment order, network capacity, preprocessing strategy, training and inference hyperparameters, and to choose a starting point for training.

Furthermore, to choose between just latent code or latent code *and* pose inference and evaluation, set the `pose_inference` parameter accordingly.

Then, run the following files in order:

1. `preprocess.py` to preprocess the shapes provided in the `mpeg7` directory.
2. `train.py` to train the model. Training can be paused and resumed by setting a starting point in `workspace.py`.
3. `reconstruct.py` to infer latent codes for the different testing sets.
4. `eval.py` to evaluate the model quantitatively or qualitatively by calling the different methods in this file.

## Other
Make sure to update the `torch` library as there was a known [issue](https://github.com/pytorch/pytorch/issues/80795) in the herein provided `arctan2` function used for determining shape orientation.