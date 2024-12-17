% Example of Gibbs sampler (Bayesian LASSO)
% Written by Dylan Green for M76X23
rng("default")
eta = .5;
N = 100;
N_M = 1000;
[A,~,~] = svd(rand(N));
physGrid = linspace(-pi,pi,N);
params.sig = 1;

f = @(X) 1/(2*eta)^2*exp(-1/eta*sum(abs(X),'all'));

fx = zeros(N,1);
fx(randi(N,[round(N/20) 1])) = 10;
fHat = A*fx + params.sig*randn(N,1);

x = zeros(N,N_M);
tausq = zeros(N,N_M);

x(:,1) = 1;
tausq(:,1) = 1;


for ii = 1:N_M
    Ginv = spdiags(1./(1/params.sig.^2 +...
        tausq(:,ii).^(-1)),0,N,N);
    Ginvchol = sqrt(Ginv.*(Ginv>0));

    x(:,ii+1) = Ginvchol * randn(N,1) + 1/params.sig.^2 * Ginv * A'*fHat;

    mu = sqrt(1./(x(:,ii+1).^2*eta^2));
    lambda = 1/eta^2;
    v = randn(N,1);
    y = v.^2;
    z = mu + mu.^2.*y/(2*lambda) - mu/(2*lambda).*sqrt(4*mu*lambda.*y+mu.^2.*y.^2);
    test_unif = rand(N,1);
    tausq(:,ii+1) = 1./z .* (test_unif<=mu./(mu+z)) + 1./(mu.^2./z) .* (test_unif>mu./(mu+z));

    %figure(1);plot(x(1,1:ii+1),x(2,1:ii+1),'.b');pause(0.1);
end

figure(1);plot(mean(x,2));hold on;
plot(lasso(A,fHat,'Lambda',params.sig^2/(N*eta),'Standardize',false));hold off;
legend('BL','Lasso');