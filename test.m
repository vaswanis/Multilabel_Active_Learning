function [Y] = test(X,W,L,phi,opts)

%noise parameters
chi = opts.chi;
small_sigma = opts.small_sigma;
%number of update iterations
maxiter = opts.maxiter;

G = X' * X;
d = size(X,2); %#(num features)
N = size(X,1); %#(num examples)
K = size(W,2); %#(num compressed labels)

%initialization
Z(1:N) = struct('mu',zeros(K,1),'sigma',eye(K));
Y(1:N) = struct('mu',zeros(L,1),'sigma',eye(L));
a0 = ones(N,L) * 1e-6;
b0 = ones(N,L) * 1e-6;

for i = 1:N
    Z(i).sigma = ((small_sigma^2 * chi^2) / (small_sigma^2 + chi^2)) * eye(K,K);
end

%test examples
concat_W = (concat_struct_attr(W,'mu'));
for i = 1:N
    
    %pause;
    x_i = X(i,:);
    Wx = small_sigma^-2 * concat_W * x_i';  

    a_i = a0(i,:) + 0.5;
    b_i = b0(i,:);

    for t = 1:maxiter
        
        %update Z(i)
        Z(i).mu = Z(i).sigma * ( Wx + ( chi^(-2) * phi * Y(i).mu)  );
        
        %find expectation of alpha_i
        %a_i = a(i,:);
        %b_i = b(i,:);
        E_alpha_i = a_i ./ b_i;
        
        %update Y(i)
	%y_old = trace(Y(i).sigma);
        Y(i).sigma = pinv(diag(E_alpha_i) + (phi' * phi) / (chi^2));
        Y(i).mu = ( Y(i).sigma * phi' * Z(i).mu ) / (chi^2);

        
        %update a,b
        b_i = b0(i,:) + 0.5 * ((diag(Y(i).sigma) + (Y(i).mu).^2))';

	%norm(y_old - trace(Y(i).sigma))

%        if norm( b_i - b_old ) < 1e-5
%		fprintf('Breaking at iteration %d\n', t);
%               break;
 %      end
        
        
        
    end
    
end

end
