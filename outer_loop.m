total_runs = 3;
L = 174;
percent_compression_list = 0:0.1:0.9;
p = size(percent_compression_list,2);
precision_table = zeros(p,total_runs);

for i = 1:p

	percent_compression = percent_compression_list(i);
	fprintf('--------------- Percent Compression = %f --------------------------------\n', percent_compression);

	for run_no = 1:total_runs

		precision_table(i,run_no) = run(L, percent_compression);
		fprintf('End of run %d\n',run_no);
	end

end
