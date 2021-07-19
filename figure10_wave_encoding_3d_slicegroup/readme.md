## Guide for Replicating Experiments in Figure 10

#### 0.   Downloading the dataset

Make sure the 3d wave-encoded dataset provided in the dropbox link is downloaded and placed into the **data/** directory.

#### 1.   Running the reconstructions

In the **wave_3d_slicegroup.ipynb** notebook, the following parameters can be set for the 3D slice-group wave-encoded reconstruction with SPARK.

**beginningSliceIndex** ~ Determines which slice group to reconstruct

**numslices_all**       ~ This value minus 1 determines the partition-encode (Rz) acceleration (the number of slices in the slice-group) 

**fovshsift**           ~ The amount of FOV shift induced by caipi sampling

**Ry**                  ~ The undersampling factor in the phase-encode (Ry) direction

**acsy**                ~ The number of phase-encode points in the (Ry) direction

Then, running the notebook will produce comparisons between cartesian SENSE, cartesian SENSE + SPARK, generalized wave-encoding, and generalized wave-encoding + SPARK.

Figure 10 was generated with the following parameters to compare reconstruction fidelity at Ry x Rz = 5 x 3: **beginningSliceIndex** = 10, **numslices_all** = 4, **fovshift** = 3, **Ry** = 5, **acsy** = 30

![Alt text](../docs/images/wave2d.png?raw=True "wave3d")
