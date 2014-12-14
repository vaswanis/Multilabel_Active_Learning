function [] = display_results(dataset)

close all;
X = 10:10:100;

load(['./Results/' dataset '_kernel_precision1_table.mat'])
load(['./Results/' dataset '_no_kernel_precision1_table.mat'])

kernel_Y = mean(kernel_precision_table,2);
kernel_Y = sort(kernel_Y);

no_kernel_Y = mean(no_kernel_precision_table,2);
no_kernel_Y = sort(no_kernel_Y);

figure(1);
plot(X,kernel_Y,'-k');
hold on;
plot(X,no_kernel_Y,'-r');


xlabel('% labels available','FontSize',24);
ylabel('Precision@1','FontSize',24);
title('Precision@1 vs Compression','FontSize',24);
legend('With Kernel', 'Without Kernel','Location','southeast');

set(gca,'FontSize',20);
saveas(gcf,['./Results/' dataset '_Precision1_vs_Compression.fig'])
print('-dpng',['./Results/' dataset '_Precision1_vs_Compression.png'])


load(['./Results/' dataset '_kernel_train_time_table.mat'])
load(['./Results/' dataset '_no_kernel_train_time_table.mat'])

kernel_Yt = mean(kernel_train_time_table,2);
kernel_Yt = sort(kernel_Yt);

no_kernel_Yt = mean(no_kernel_train_time_table,2);
no_kernel_Yt = sort(no_kernel_Yt);

figure(2);
plot(X,kernel_Yt,'-k');
hold on;
plot(X,no_kernel_Yt,'-r');

xlabel('% labels available','FontSize',24);
ylabel('Training Time','FontSize',24);
title('Training Time vs Compresseion','FontSize',24);
legend('With Kernel', 'Without Kernel','Location','southeast');

set(gca,'FontSize',20);
saveas(gcf,['./Results/' dataset '_Training_Time_vs_Compression.fig']);
print('-dpng',['./Results/' dataset '_Training_Time_vs_Compression.png']);

end

