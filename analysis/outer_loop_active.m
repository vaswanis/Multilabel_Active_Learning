%number of runs 
total_runs = 1;

%adding paths
addpath('../model/');
addpath('../utils/');

%load dataset
dataset = 'enron';
load(dataset);

%artfiicailly reduce number of ssamples in dataset
%X = X(1:500,:);
%y = y(1:500,:);

%compression ratio
percent_compression = 0.5;

%general parameters
opts.train_maxiter = 200;
opts.test_maxiter = 100;

opts.train_fraction = 0.8;
opts.CV = 0;
opts.k = 1;

%active learning parameters
opts.max_rounds = 100;
opts.selection_batch_size = 1;
opts.N_train_initial = 200;

no_kernel_uncertainty_precision_table = zeros(opts.max_rounds,total_runs);
no_kernel_random_precision_table = zeros(opts.max_rounds,total_runs);
kernel_uncertainty_precision_table = zeros(opts.max_rounds,total_runs);
kernel_random_precision_table = zeros(opts.max_rounds,total_runs);

%number of true labels
L = size(y,2);

%number of compressed labels
K = floor((1 - percent_compression) * L);

fprintf('--------------- Percent Compression = %f --------------------------------\n', percent_compression);

for run_no = 1:total_runs

	%random projection matrix
	phi = rand(K,L);

	%without kernelization
	opts.kernelize = 0;
        [precision_uncertainty, precision_random] = run_active(percent_compression,X,y,phi,opts);
	no_kernel_uncertainty_precision_table(:,run_no) = precision_uncertainty;
	no_kernel_random_precision_table(:,run_no) = precision_random;


	%with kernelization
	opts.kernelize = 1;
       [precision_uncertainty, precision_random] = run_active(percent_compression,X,y,phi,opts);
	kernel_uncertainty_precision_table(:,run_no) = precision_uncertainty;
	kernel_random_precision_table(:,run_no) = precision_random;

	fprintf('End of run %d\n',run_no);

end

no_kernel_mean_uncertainty_precision = mean(no_kernel_uncertainty_precision_table,2)
no_kernel_mean_random_precision = mean(no_kernel_random_precision_table,2)

kernel_mean_uncertainty_precision = mean(kernel_uncertainty_precision_table,2)
kernel_mean_random_precision = mean(kernel_random_precision_table,2)


save([dataset '_active_kernel_uncertainty_precision1_table'], 'kernel_uncertainty_precision_table' );
save([dataset '_active_kernel_random_precision1_table'], 'kernel_random_precision_table' );
save([dataset '_active_no_kernel_uncertainty_precision1_table'], 'no_kernel_uncertainty_precision_table' );
save([dataset '_active_no_kernel_random_precision1_table'], 'no_kernel_random_precision_table' );





