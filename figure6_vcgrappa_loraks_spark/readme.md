## Guide to performing Experiments for Figure 6

#### 1.   Preparing data for SPARK

To prepare data for virtual-coil GRAPPA, consider the script **prep_vgrappa_forspark.m"**.  Adjust the variable **Rx** to the desired 1D acceleration.  Then, running the script will generate and save the necessary data for SPRAK.  

Following an analagous procedure using the script **prep_loraks_forspark.m** will generate the appropriate data for LORAKS with SPARK.

#### 2.   Performing the SPARK Correction

Running the notebooks **loraks_spark.ipynb** and **svcgrappa_spark.ipynb** will generate and save results when applying SPARK to LORAKS and virtual-coil GRAPPA reconstructions respectively.

### Figure in Manuscript

Following the above procedure while varying **Rx** produces the results for Figure 6 in the manuscript (with a comparison to a couple of residual-RAKI reconstructions):

![Alt text](../docs/images/potential_figure_R5R6_rraki_loraks_vc.png?raw=True "loraksvc")
