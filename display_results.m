function [] = display_results(dataset)

close all;
X = 10:10:100;

load(['./Results/' dataset '_precision1_table.mat'])

Y = mean(precision_table,2);
Y = sort(Y);
figure(1);
plot(X,Y);
xlabel('% labels available','FontSize',24);
ylabel('Precision@1','FontSize',24);
title('Precision@1 vs Compression','FontSize',24);
set(gca,'FontSize',20);
saveas(gcf,['./Results/' dataset '_Precision1_vs_Compression.fig'])
print('-dpng',['./Results/' dataset '_Precision1_vs_Compression.png'])


load(['./Results/' dataset '_train_time_table.mat'])

Yt = mean(train_time_table,2);
Yt = sort(Yt);
figure(2);
plot(X,Yt);
xlabel('% labels available','FontSize',24);
ylabel('Training Time','FontSize',24);
title('Training Time vs Compresseion','FontSize',24);
set(gca,'FontSize',20);
saveas(gcf,['./Results/' dataset '_Training_Time_vs_Compression.fig']);
print('-dpng',['./Results/' dataset '_Training_Time_vs_Compression.png']);

end

