% Perform my own comparison of RAKI vs GRAPPA vs Residual RAKI vs SPARK at a variety of ACS
% sizes and accelerations
close all

load('raki_ablation.mat')
load('spark_ablation.mat')

load('residual_raki_ablation.mat')
rraki_rmse = reshape(rraki_rmse,5,2).'/100;
all_rraki  = permute(reshape(all_rraki,236,188,5,2),[1,2,4,3]);


%% Plot all rmses for each acceleration (each acceleration = a different plot)
A = length(accelerations);
S = length(acs_sizes);
start = 1;

lw = 10;
%-Seperate plot for each acceleration
for aa = 1:A
    figure; hold on;
    plot(acs_sizes(start:end),all_grappa_rmse(aa,start:end).'*100,'-s','MarkerSize',20,'LineWidth',lw);
    plot(acs_sizes(start:end),all_raki_rmse(aa,start:end).'*100,'-s','MarkerSize',20,'LineWidth',lw);
    plot(acs_sizes(start:end),rraki_rmse(aa,start:end).'*100,'-s','MarkerSize',20,'LineWidth',lw);
    plot(acs_sizes(start:end),all_spark_rmse(aa,start:end).'*100,'-s','MarkerSize',20,'LineWidth',lw);

    if(aa == 1)
        ylim([5,10])
    end
    
    if(aa == 2)
        ylim([5,15])
    end
    ax = gca; ax.XColor = [1,1,1]; ax.YColor = [1,1,1]; set(gca,'FontSize',30) 
    fig = gcf; fig.Color = 'black'; fig.InvertHardcopy = 'off';
    legend({'PI','RAKI','res-RAKI','SPARK'},'FontSize',15)
end

%% Show comparisons at R = 6 with 30 or 36 ACS lines
ss = 3;  %acs index
aa = 1;  %acceleration index
nr = @(x) x / max(abs(x(:)));
n  = @(x,aa,ss) nr(abs(squeeze(x(aa,ss,:,:)).'));

fprintf('acs size: %d\n',acs_sizes(ss));
fprintf('accel:    %d\n',accelerations(aa));
fprintf('rmse values:\n')
fprintf('  grappa: %.2f\n',all_grappa_rmse(aa,ss)*100);
fprintf('  raki:   %.2f\n',all_raki_rmse(aa,ss)*100);
fprintf('  rraki:  %.2f\n',rraki_rmse(aa,ss)*100);
fprintf('  spark:  %.2f\n',all_spark_rmse(aa,ss)*100);

%-Show the images
images = [n(all_grappa,aa,ss),n(all_spark,aa,ss),nr(all_raki(:,:,aa,ss).'),nr(all_rraki(:,:,aa,ss))];

errors = abs([n(all_grappa,aa,ss) - nr(truth),...
                    n(all_spark,aa,ss) - nr(truth),...
                    nr(all_raki(:,:,aa,ss).') - nr(truth),...
                    nr(all_rraki(:,:,aa,ss)) - nr(truth)])*10;

display = [images;errors];

figure; imshow(display)

imwrite(display,sprintf('image_error_R%d_acs%d.png',accelerations(aa),acs_sizes(ss)));
