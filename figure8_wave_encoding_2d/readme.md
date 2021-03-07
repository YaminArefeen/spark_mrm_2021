## Guide for Replicating Experiments in Figure 8

#### 0.   Downloading the dataset

Make sure the 2d wave-encoded dataset provided in the dropbox link is downloaded and placed into the **data/** directory.

#### 1.   Running the reconstructions

In the **wave_2d_spark.ipynb** notebook, set the **Ry** variable to the desired acceleration to be evaluated and **acsy** to the number of phase-encode points to be  sampled.  Then, running the entire script will produce comparisons between Cartesian SENSE, Cartesian SENSE + SPARK, generalized SENSE with wave-encoding, and generalized SENSE with wave-encoding + SPARK.  

Running the notebook for **Ry = 5,6** produces the results in Figure 8:

![Alt text](../docs/images/wave2d.png?raw=True "wave2d")
