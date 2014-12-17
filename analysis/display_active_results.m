function [] = display_active_results( dataset )

no_points = 30;

load([dataset '_active_uncertainty_precision1.mat']);
load([dataset '_active_random_precision1.mat'])

figure(1);
plot(1:no_points,precision_uncertainty(1:no_points),'-k');
hold on;
plot(1:no_points,precision_random(1:no_points),'-r');

xlabel('Number of points added','FontSize',24);
ylabel('Precision@1','FontSize',24);
title('Precision@1 vs Number of points added','FontSize',24);
legend('Uncertainty Sampling', 'Random Sampling','Location','northeast');

set(gca,'FontSize',20);
saveas(gcf,['../results/' dataset '_Precision1_vs_Number_of_points_added.fig'])
print('-dpng',['../results/' dataset 'Precision1_vs_Number_of_points_added.png'])


end

