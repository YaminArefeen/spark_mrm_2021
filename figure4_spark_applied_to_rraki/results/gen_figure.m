%% Summarizing results where we attempt combine SPARK and rraki for a variety of ACS sizes and accelerations
load('residual_raki_with_spark_ablation.mat')

[N,M] = size(truth);
A = length(accelerations);
S = length(acs_sizes);

rraki_acs = permute(reshape(all_rraki_acs,N,M,S,A),[2,1,4,3]);
spark     = permute(reshape(all_spark,N,M,S,A),[2,1,4,3]);
full      = truth.';

%-Computing rmses
rmse_spark = zeros(A,S);
rmse_rraki = zeros(A,S);

for aa = 1:A
    for ss = 1:S
        rmse_spark(aa,ss) = rmse(full,spark(:,:,aa,ss)) * 100;
        rmse_rraki(aa,ss) = rmse(full,rraki_acs(:,:,aa,ss)) * 100;
    end
end
%% Plot all rmses for each acceleration (each acceleration = a different plot)
start = 1;

lw = 10;
%-Seperate plot for each acceleration
for aa = 1:A
    figure; hold on;
    plot(acs_sizes(start:end),rmse_rraki(aa,start:end).','-s','MarkerSize',20,'LineWidth',lw);
    plot(acs_sizes(start:end),rmse_spark(aa,start:end).','-s','MarkerSize',20,'LineWidth',lw);

    ax = gca; ax.XColor = [1,1,1]; ax.YColor = [1,1,1]; set(gca,'FontSize',30) 
    fig = gcf; fig.Color = 'black'; fig.InvertHardcopy = 'off';
    legend({'residual-RAKI','residual-RAKI + SPARK'},'FontSize',15)
end

%% Show images for each acceleration at a particular acs_size
ss = 4;  %acs index
aa = 1;  %acceleration index

cr = 30;
nr = @(x) x / max(abs(x(:)));
n  = @(x,aa,ss) nr(abs(squeeze(x(cr:end,:,aa,ss))));

fprintf('acs size: %d\n',acs_sizes(ss));
fprintf('accel:    %d\n',accelerations(aa));
fprintf('rmse values:\n')
fprintf('  rraki:   %.2f\n',  rmse_rraki(aa,ss));
fprintf('  spark:   %.2f\n',  rmse_spark(aa,ss));

%-Show the images
display = [n(rraki_acs,aa,ss),n(spark,aa,ss);...
   abs([n(rraki_acs,aa,ss) - nr(full(cr:end,:)),n(spark,aa,ss) - nr(full(cr:end,:))])*10];
    
figure; imshow(display)

imwrite(display,'results.png');