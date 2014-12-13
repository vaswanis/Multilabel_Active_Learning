%number of runs 
total_runs = 1;

%load dataset
load './datasets/enron'

%artfiicailly reduce number of ssamples in dataset
%X = X(1:500,:);
%y = y(1:500,:);

%compression ratios to run for
%percent_compression_list = 0:0.1:0.9;
percent_compression_list = 0.5;

%general parameters
opts.train_maxiter = 200;
opts.test_maxiter = 100;

opts.train_fraction = 0.8;
opts.CV = 0;
opts.k = 1;
opts.kernelize = 1;

%active learning parameters
opts.max_rounds = 50;
opts.selection_batch_size = 1;
opts.N_train_initial = 200;

p = size(percent_compression_list,2);
precision_table = zeros(p,total_runs);

for i = 1:p

	percent_compression = percent_compression_list(i);
	fprintf('--------------- Percent Compression = %f --------------------------------\n', percent_compression);

	for run_no = 1:total_runs
		[precision_uncertainty, precision_random] = run_active(percent_compression,X,y,opts);
		fprintf('End of run %d\n',run_no);
	end

end


