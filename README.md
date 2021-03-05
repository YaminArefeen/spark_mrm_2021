# Code to Replicate Figures in "Scan-Specific Artifact Reduction in K-space (SPARK) Neural Networks Syngerize with Physics-based Reconstruction to Accelerate "MRI" 

Written by Yamin Arefeen.  Please feel free to contact me and ask questions through email at yarefeen@mit.edu.

SPARK trains a convolutional-neural-network to estimate k-space errors made by an input reconstruction technique by back-propagating from the mean-squared-error loss between an auto-calibration signal (ACS) and the input technique’s reconstructed ACS.  In this repository, we provide the code and data used to generate Figures 4-9.  The associated data for Figures 6-9 can be downloaded from https://www.dropbox.com/sh/zveq2tfh7mgr9qk/AABSuSM23QOFVAe0SJ9oBIm6a?dl=0.  

Self-contained code and instructions to generate each of the figures can be found in the folders for each figure.

## Figure 1: Pseudo-replica esque Technique for Evaluating SPARK and GRAPPA

Applies a process similar to the pseudo replica technique desribed in "Comprehensive Quantification of Signal-to-Noise Ratio and g-Factor for Image-Based and k-Space-Based Parallel Imaging Reconstructions" to compare GRAPPA and SPARK reconstructions on a single axial slice from an MPRAGE acquisition.  

![Alt text](docs/images/noisemap_gfactor_attempt.png?raw=True "pseudo-relica")

## Figure 2: Spark with LORAKS and VC GRAPPA

SPARK applied to LORAKS and virtual-coil GRAPPA on a single axial slice from an MPRAGE acquisition; demonstrating how SPARK synergizes with advanced 2D parallel imaging techniques.

![Alt text](docs/images/potential_figure_R5R6_loraks_vc.png?raw=True "loraksvc")
