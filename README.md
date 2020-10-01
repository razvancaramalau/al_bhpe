# Active Learning for Bayesian 3D Hand Pose Estimation

The code requires pre-processed hand detected and cropped images for the 3D Hand Pose Dataset: ICVL, NYU, BigHand2.2;
This should be place in the "data/source/" directory 
The pre-processed dataset can be made available upon request.
Before running, please check and tune the parameters in the config.py file and at the beginning of the main.py function.

The attached videos contain testing images from the three Hand Pose benchmarks. An extended parallel qualitative comparison
is shown of the Bayesian DeepPrior after 10 active learning sampling stages. We attach the two analysed uncertainty variances
to the joint locations. Due to the limited uploading size, we skipped frames for NYU and BigHand2.2 dataset.
