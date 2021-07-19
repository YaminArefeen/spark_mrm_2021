## Guide to performing experiments for Figure 3

#### 1. Preparing GRAPPA reconstructions for SPARK

Run the Matlab script **gen_grappa_recons.m** to perform GRAPPA reconstructions at the varying acs-sizes and acceleration rates.

#### 2. Applying the SPARK correction

Run the jupyter notebook **ablation_spark.ipynb** to apply the SPARK corrections to the generated GRAPPA reconstructions

#### 3. RAKI Reconstructions

Run the jupyter notebook **ablation_raki.ipynb** to perform RAKI reconstructions at the same acs sizes and acceleration rates

#### 4. residual-RAKI reconstruction

Run the jupyter notebook **residual_raki_ablation.ipynb** to perform residual-RAKI reconstructions at the same acs sizes and acceleration rates

#### 5. Displaying comparisons

Run the script **gen_figure.m** in the directory **results/genfigure.m** to generate the figures.

### Figure in manuscript

![Alt text](docs/images/residual_raki_comparison.png?raw=True "spark_raki_rraki")

