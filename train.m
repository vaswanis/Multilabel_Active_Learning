function [W] = train(X,y,K,opts)

%K = 10; %#(compressed labels)

L = size(y,2); %#(true labels)

%noise parameters
chi = opts.chi;
small_sigma = opts.small_sigma;
%number of update iterations
maxiter = opts.maxiter;

G = X' * X;

d = size(X,2);%#(num features)
N = size(X,1); %#(num examples)

%random projection matrix
phi = rand(K,L);


Y_sample_mean = mean(y)';

%Z matrix
z = (phi * y')';

Z_sample_mean = mean(z)';

%initialization
W(1:K) = struct('mu',zeros(d,1),'sigma',eye(d));
Z(1:N) = struct('mu',zeros(K,1),'sigma',eye(K));
Y(1:N) = struct('mu',Y_sample_mean,'sigma',cov(y));
a = ones(N,L) * 1e-6;
b = ones(N,L) * 1e-6;


%update equations
a0 = a;
b0 = b;

% for i = 1:N
%     Z(i).sigma = (small_sigma^2 + chi^2) * eye(K,K);
% end
for k = 1:K
    W(k).sigma = pinv(small_sigma^(-2) * G + eye(d));
end

%train examples
for t = 1:maxiter
t    
    for i = 1:N
        
        x_i = X(i,:);
        y_i = y(i,:);
%         z_i = z(i,:);
        
        %find expectation of alpha_i
%         a_i = a(i,:);
%         b_i = b(i,:);
%         E_alpha_i = a_i ./ b_i;
        
        %update Y(i)
%         Y(i).sigma = pinv(diag(E_alpha_i) + (phi' * phi) / (chi^2));
%         Y(i).mu = ( Y(i).sigma * phi' * Z(i).mu ) / (chi^2);
        
        %update a,b
%         a = a0 + 0.5;
%         b(i,:) = b0(i,:) + 0.5 * ((diag(Y(i).sigma) + (Y(i).mu).^2))';
        
        %update Z(i)
        Z(i).mu = Z(i).sigma * ( (small_sigma^-2 * (concat_struct_attr(W,'mu') * x_i')) + ( chi^(-2) * phi * y_i')  );
        
    end
    
    for k = 1:K
        %update W(i)
        temp = (concat_struct_attr(Z,'mu'));
        W(k).mu = (small_sigma)^(-2) * W(k).sigma * X' * temp(:,k);
    end
    norm(W(1).mu)
    
    
    
end

end


