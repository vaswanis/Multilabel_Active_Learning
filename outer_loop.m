%number of runs 
total_runs = 3;

%load dataset
load './datasets/nuswide'
X = X(1:2000,:);
y = y(1:2000,:);

%compression ratios to run for
percent_compression_list = 0:0.1:0.9;
%percent_compression_list = 0.8;

%parameters
opts.maxiter = 500;
opts.train_fraction = 0.8;
opts.CV = 0;
opts.k = 1;

p = size(percent_compression_list,2);

precision_table = zeros(p,total_runs);
train_time_table = zeros(p,total_runs);

for i = 1:p

	percent_compression = percent_compression_list(i);
	fprintf('--------------- Percent Compression = %f --------------------------------\n', percent_compression);

	for run_no = 1:total_runs

	        [precision,train_time,test_time]= run(percent_compression,X,y,opts);

		precision_table(i,run_no) = precision;
		train_time_table(i,run_no) = train_time;
		

		fprintf('End of run %d\n',run_no);
	end

end

mean_precision = mean(precision_table,2)
mean_training_time = mean(train_time_table,2)
