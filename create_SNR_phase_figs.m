clear all
close all

sparse_domain = 'transform';
forward_op = ["fft" "blur" "rand"];
savefolder = '/Users/dylangreen/Documents/MATLAB/BayesianLASSO/figures/N200x1';
percent_zero = 0.2;

%%
for ii = 1:length(forward_op)
    savefile = strcat('ErrorCompare_',forward_op(ii),'.png');
    loadfile = strcat('/Users/dylangreen/Documents/MATLAB/BayesianLASSO/figures/N200x1/DistTrueSol',forward_op(ii),'.mat');
    load(loadfile);

    figure(ii);plot(SNRtotal,log(mlePhaseDistFromTrue),'r','LineWidth',2);hold on
    plot(SNRtotal,log(avgPhaseDistFromTrue),'b','LineWidth',2);hold off
    xlabel('SNR','Interpreter','latex');ylabel('Average Error (logarithmic scale)','Interpreter','latex')

    set(gcf,'Position',[100 100 500 300]);
    saveas(gcf,strcat(savefolder,filesep,savefile))
end
legend('Generalized LASSO','CVBL','Interpreter','latex')

%%
figure(3);xlabel([]);ylabel([]);
% Call the legend to your choice, I used a horizontal legend here
legend_handle = legend('Orientation','vertical');
% Set the figure Position using the normalized legend Position vector
% as a multiplier to the figure's current position in pixels
% This sets the figure to have the same size as the legend
set(gcf,'Position',(get(legend_handle,'Position')...
    .*[0, 0, 1, 1].*get(gcf,'Position')));
% The legend is still offset so set its normalized position vector to
% fill the figure
set(legend_handle,'Position',[0,0,1,1]);
% Put the figure back in the middle screen area
set(gcf, 'Position', get(gcf,'Position') + [500, 400, 0, 0]);
savelegend = strcat('legend_ErrorCompare.png');
saveas(gcf,strcat(savefolder,filesep,savelegend));

%%
savefile = strcat('SNRcompare',sparse_domain,'.png');
for ii = 1:length(forward_op)
    if strcmp(forward_op(ii),"rand")
        loadfile = strcat('/Users/dylangreen/Documents/MATLAB/BayesianLASSO/figures/N200x1/SNRcompare_',sparse_domain,'_',forward_op(ii),'_percZero',num2str(100*percent_zero),'.mat');
    else
        loadfile = strcat('/Users/dylangreen/Documents/MATLAB/BayesianLASSO/figures/N200x1/SNRcompare_',sparse_domain,'_',forward_op(ii),'.mat');
    end
    load(loadfile);

    figure(4);hold on;plot(SNRtotal,log(CVBL_error),'LineWidth',2);hold off
end
xlabel('SNR','Interpreter','latex');ylabel('Average Error (logarithmic scale)','Interpreter','latex')
set(gcf,'Position',[100 100 500 300]);
saveas(gcf,strcat(savefolder,filesep,savefile));
legend('$F_F$','$F_B$','$F_U$','Interpreter','latex')

%%
figure(4);xlabel([]);ylabel([]);
% Call the legend to your choice, I used a horizontal legend here
legend_handle = legend('Orientation','vertical');
% Set the figure Position using the normalized legend Position vector
% as a multiplier to the figure's current position in pixels
% This sets the figure to have the same size as the legend
set(gcf,'Position',(get(legend_handle,'Position')...
    .*[0, 0, 1, 1].*get(gcf,'Position')));
% The legend is still offset so set its normalized position vector to
% fill the figure
set(legend_handle,'Position',[0,0,1,1]);
% Put the figure back in the middle screen area
set(gcf, 'Position', get(gcf,'Position') + [500, 400, 0, 0]);
savelegend = strcat('legend_SNR.png');
saveas(gcf,strcat(savefolder,filesep,savelegend));

%%
SNRused = "20";

savefile = strcat('UnderSampleCompare',sparse_domain,'.png');
loadfile = strcat('/Users/dylangreen/Documents/MATLAB/BayesianLASSO/figures/N200x1/underSampleDistTrueSol_SNR',SNRused,'.mat');
load(loadfile);

figure(5);plot(1-percent_zero_total,log(mlePhaseDistFromTrue),'r','LineWidth',2);hold on
plot(1-percent_zero_total,log(avgPhaseDistFromTrue),'b','LineWidth',2);hold off

xlabel('$\nu$','Interpreter','latex');ylabel('Average Error (logarithmic scale)','Interpreter','latex')
% legend('Generalized LASSO','CVBL','Interpreter','latex','Location','northeast')
set(gcf,'Position',[100 100 500 300]);
saveas(gcf,strcat(savefolder,filesep,savefile));

%%
figure(5);xlabel([]);ylabel([]);
% Call the legend to your choice, I used a horizontal legend here
legend_handle = legend('Orientation','vertical');
% Set the figure Position using the normalized legend Position vector
% as a multiplier to the figure's current position in pixels
% This sets the figure to have the same size as the legend
set(gcf,'Position',(get(legend_handle,'Position')...
    .*[0, 0, 1, 1].*get(gcf,'Position')));
% The legend is still offset so set its normalized position vector to
% fill the figure
set(legend_handle,'Position',[0,0,1,1]);
% Put the figure back in the middle screen area
set(gcf, 'Position', get(gcf,'Position') + [500, 400, 0, 0]);
savelegend = strcat('legend_SNR.png');
saveas(gcf,strcat(savefolder,filesep,savelegend));