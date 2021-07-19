## Guide to performing experiments for Figure 5:

#### 1.     Preparing data for SPARK

In the Matlab script **psuedo_replica_forspark.m**, set the variable **Rx** to the desired 1D acceleration factor to be evaluated.  Then running *psuedo_replica_forspark.m*, will generate and save all of the necessary GRAPPA reconstructions associated with the technique.

#### 2.     Including the SPARK corrections

In the jupyter notebook **spark_pseudoreplica.ipynb** set the variable **accel** to the same value of **Rx** set in the previous step.  Then, running the entire notebook will generate and save all of the necessary SPARK corrections and reconstructions.


#### 3.     Evaluating GRAPPA and SPARK

In the Matlab script **evaluate_snr_proxy.m** set the variable **accell** to the same values as in the previous steps.  Then, running the entire script will generate maps comparing GRAPPA and SPARK which can be thought of as a proxy for SNR.

### Figure in Manuscript

Using these three scripts and varying the chosen acceleration to **Rx = 4,5,6** produced Figure 4 below:

![Alt text](../docs/images/noisemap_gfactor_attempt.png?raw=True "pseudo-relica")


