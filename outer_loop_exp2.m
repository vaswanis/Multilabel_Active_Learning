%number of runs 
total_runs = 5;

%load dataset
dataset = 'enron';
load(['./datasets/' dataset]);

%artificially reduce number of samples in dataset
%X = X(1:500,:);
%y = y(1:500,:);

%compression ratios to run for
%percent_compression_list = 0:0.1:0.9;
percent_compression_list = [0.1,0.3,0.5,0.7,0.9];

%parameters
opts.train_maxiter = 200;
opts.test_maxiter = 100;

opts.train_fraction = 0.8;
opts.CV = 0;
opts.k = 1;

p = size(percent_compression_list,2);

precision_table = zeros(p,total_runs);
train_time_table = zeros(p,total_runs);

%number of true labels
L = size(y,2);

%types of phi
%phi1 - random
%phi2 - random binary
%phi3 - dropout

for i = 1:p

	percent_compression = percent_compression_list(i);

	%number of compressed labels
	K = floor((1 - percent_compression) * L);

	fprintf('--------------- Percent Compression = %f --------------------------------\n', percent_compression);

	for run_no = 1:total_runs

		%dropping out labels 
		phi = rand(K,L);
		for j = 1:(L-K)
			col = ceil(rand() * L);  
			phi(:,col) = zeros(K,1);
		end

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

save([dataset '_phi3_kernel_precision1_table'], 'kernel_precision_table' );
save([dataset '_phi3_kernel_train_time_table'], 'kernel_train_time_table' );
save([dataset '_phi3_no_kernel_precision1_table'], 'no_kernel_precision_table' );
save([dataset '_phi3_no_kernel_train_time_table'], 'no_kernel_train_time_table' );

