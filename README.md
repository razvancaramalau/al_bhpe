# Active Learning for Bayesian 3D Hand Pose Estimation
If you find this code useful please cite: https://arxiv.org/abs/2010.00694

## Proposed pipeline
![alt text](https://github.com/razvancaramalau/al_bhpe/blob/master/pipeline.png?raw=true)


The code requires pre-processed hand detected and cropped images for the 3D Hand Pose Dataset: ICVL, NYU, BigHand2.2;
This should be place in the "data/source/" directory 
The pre-processed dataset can be made available upon request(r.caramalau18@imperial.ac.uk).

Before running, please check and tune the parameters in the config.py file and at the beginning of the main.py function.

## Bayesian DeepPrior results with aleatoric and epistemic uncertainty
![alt text](https://github.com/razvancaramalau/al_bhpe/blob/master/poster.png?raw=true)

## Qualitative analysis on the ICVL, NYU, BigHand2.2M datasets
https://www.youtube.com/playlist?list=PLYk-b7Fzd073O2ksuTSvxLNXiMEOMSaIh

## Required Python3 libraries:
```bash 

numpy
matplotlib
tensorflow
keras
opencv
skicit-learn
scipy
math
h5py
glob
gc
argparse
```

