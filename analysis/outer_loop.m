%number of runs 
total_runs = 1;

%adding paths
addpath('../model/');
addpath('../utils/');
addpath('../datasets/');

%load dataset
dataset = 'CAL500';
load(dataset);

%artificially reduce number of samples in dataset
%X = X(1:500,:);
%y = y(1:500,:);

%compression ratios to run for
%percent_compression_list = 0:0.1:0.9;
percent_compression_list = 0.9;

%parameters
opts.train_maxiter = 200;
opts.test_maxiter = 100;

opts.train_fraction = 0.9;
opts.CV = 1;
opts.k = 1;

opts.kernelize = 1;

small_sigma_values = [1e-2, 1e-1];
chi_values = [1e-5, 1e-4, 1e-3];
kernel_length_scale_values = [1 10];

n = 2;
if opts.CV
    [best_small_sigma, best_chi, best_kernel_length_scale] = cross_validation(percent_compression, X, y, phi, opts, n, small_sigma_values, chi_values, kernel_length_scale_values)
else
    best_small_sigma = 1e-2;
    best_chi = 1e-4;
    best_kernel_length_scale = 1;
end

opts.small_sigma = best_small_sigma;
opts.chi = best_chi;
opts.kernel_length_scale = best_kernel_length_scale;

p = size(percent_compression_list,2);

precision_table = zeros(p,total_runs);
train_time_table = zeros(p,total_runs);

%number of true labels
L = size(y,2);

for i = 1:p

	percent_compression = percent_compression_list(i);

	%number of compressed labels
	K = floor((1 - percent_compression) * L);

	fprintf('--------------- Percent Compression = %f --------------------------------\n', percent_compression);

	for run_no = 1:total_runs

		%random projection matrix
		phi = rand(K,L);

		%without kernelization
		opts.kernelize = 0;
	        [precision,train_time,test_time]= run(percent_compression,X,y,phi,opts);
		no_kernel_precision_table(i,run_no) = precision;
		no_kernel_train_time_table(i,run_no) = train_time;


		%with kernelization
		opts.kernelize = 1;
	        [precision,train_time,test_time]= run(percent_compression,X,y,phi,opts);
		kernel_precision_table(i,run_no) = precision;
		kernel_train_time_table(i,run_no) = train_time;

		fprintf('End of run %d\n',run_no);

	end
end

no_kernel_mean_precision = mean(no_kernel_precision_table,2)
no_kernel_mean_train_time = mean(no_kernel_train_time_table,2)

kernel_mean_precision = mean(kernel_precision_table,2)
kernel_mean_train_time = mean(kernel_train_time_table,2)

save([dataset '_kernel_precision1_table'], 'kernel_precision_table' );
save([dataset '_kernel_train_time_table'], 'kernel_train_time_table' );

save([dataset '_no_kernel_precision1_table'], 'no_kernel_precision_table' );
save([dataset '_no_kernel_train_time_table'], 'no_kernel_train_time_table' );

