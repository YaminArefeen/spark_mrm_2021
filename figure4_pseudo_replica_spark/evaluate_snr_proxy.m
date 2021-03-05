%%
% Given the results from "spark_pseudoreplica.ipynb", generate the associated pseudo-replica SNR maps
% from the GRAPPA and SPARK + GRAPPA reconstructions

%% Load the results
accel = 6;  %Acceleration factor to be analyzed
load(sprintf('results/results_Rx%dRy1.mat',accel))

%% Generate the maps
grappa_replicas = permute(grappa_montecarlo,[2,3,1]);
spark_replicas  = permute(spark_montecarlo,[2,3,1]);

std_grappa = std(real(grappa_replicas),[],3);
std_spark  = std(real(spark_replicas),[],3);

map_grappa = abs(baseline_grappa)./std_grappa;
map_spark  = abs(baseline_spark)./std_spark;

map_grappa(isnan(map_grappa)) = 0;
map_spark(isnan(map_spark))   = 0;

%% Display the maps
cmin = 0; cmax = 50;
figure; imagesc([map_grappa,map_spark]); axis image; colorbar; caxis([cmin,cmax]);