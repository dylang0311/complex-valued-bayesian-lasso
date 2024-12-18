function [a,b,tausq,etasq] = sparse_mag_lasso(fHat,A,AH,params,unitary,sigStDev,forward_op)
a = zeros(params.N1*params.N2,params.N_M);
b = zeros(params.N1*params.N2,params.N_M);
tausq = ones(params.N1*params.N2,params.N_M);
etasq = zeros(1,params.N_M);

a(:,1) = real(AH(fHat));
b(:,1) = imag(AH(fHat));
tausq(:,1) = ones(params.N1*params.N2,1);
etasq(1) = 0.1;

if ~unitary
    if strcmp(forward_op,"blur")
        bAmat = sparse([-imag(A(eye(params.N1*params.N2)));real(A(eye(params.N1*params.N2)))]);
        aAmat = sparse([real(A(eye(params.N1*params.N2)));imag(A(eye(params.N1*params.N2)))]);
    elseif strcmp(forward_op,"rand")
        bAmat = @(x) [-imag(A(x));real(A(x))];
        aAmat = @(x) [real(A(x));imag(A(x))];

        bAmatH = @(x) imag(AH(x(1:end/2)))+real(AH(x(end/2+1:end)));
        aAmatH = @(x) real(AH(x(1:end/2)))-imag(AH(x(end/2+1:end)));
    end
    fHat_real = [real(fHat);imag(fHat)];
end

for kk = 1:params.N_M
    if kk < 10
        tic
    end
    
    Ginv = spdiags(1./(2/sigStDev.^2 +...
        tausq(:,kk).^(-1)),0,params.N1*params.N2,params.N1*params.N2);
    Ginvchol = sqrt(Ginv.*(Ginv>0));

    if unitary
        a(:,kk+1) = Ginvchol*randn(params.N1*params.N2,1) + 2/sigStDev.^2 * Ginv * real(AH(fHat - A(1i*b(:,kk))));
    
        b(:,kk+1) = Ginvchol*randn(params.N1*params.N2,1) + 2/sigStDev.^2 * Ginv * imag(AH(fHat - A(a(:,kk+1))));
    elseif strcmp(forward_op,"blur")
        mu_a = fHat_real - bAmat*(b(:,kk));
        a(:,kk+1) = perturbation_optimization_sampler(mu_a,tausq(:,kk),sigStDev,kk,params,aAmat,forward_op,1);

        mu_b = fHat_real - aAmat*(a(:,kk+1));
        b(:,kk+1) = perturbation_optimization_sampler(mu_b,tausq(:,kk),sigStDev,kk,params,bAmat,forward_op,1);
    elseif strcmp(forward_op,"rand")
        mu_a = fHat_real - bAmat(b(:,kk));
        a(:,kk+1) = perturbation_optimization_sampler(mu_a,tausq(:,kk),sigStDev,kk,params,aAmat,forward_op,1,aAmatH);

        mu_b = fHat_real - aAmat(a(:,kk+1));
        b(:,kk+1) = perturbation_optimization_sampler(mu_b,tausq(:,kk),sigStDev,kk,params,bAmat,forward_op,1,bAmatH);
    end

    mu = sqrt(1./((a(:,kk+1).^2+b(:,kk+1).^2)*etasq(kk)));
    lambda = 1/etasq(kk);
    v = randn(params.N1*params.N2,1);
    y = v.^2;
    x = mu + mu.^2.*y/(2*lambda) - mu/(2*lambda).*sqrt(4*mu*lambda.*y+mu.^2.*y.^2);
    test_unif = rand(params.N1*params.N2,1);
    tausq(:,kk+1) = 1./x .* (test_unif<=mu./(mu+x)) + 1./(mu.^2./x) .* (test_unif>=mu./(mu+x));

    etasq(kk+1) = 1/gamrnd((params.N1*params.N2)+params.lambda_r,1/(sum(tausq(:,kk+1)/2)+params.lambda_delta));
    
    if kk < 10
        toc
    end
end
a = a(:,params.B:end);
b = b(:,params.B:end);
tausq = tausq(:,params.B:end);
etasq = etasq(params.B:end);
end