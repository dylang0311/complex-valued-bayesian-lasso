function [g,phi,tausq,etasq] = sparse_transf_lasso(fHat,Ainv,A,AH,params,unitary,sigStDev,forward_op,periodicBC,fx)
g = zeros(params.N1*params.N2,params.N_M);
phi = zeros(params.N1*params.N2,params.N_M);

PAORDER = 1;
L = sparse_operator(params,1,PAORDER,periodicBC);
tausq = ones(size(L,1),params.N_M);
etasq = zeros(1,params.N_M);
post_density = zeros(1,params.N_M);

g(:,1) = ones(params.N1*params.N2,1);
phi(:,1) = ones(params.N1*params.N2,1);
tausq(:,1) = 1;
etasq(1) = 1;

if params.learn_sigma
    sigsq = .0001*ones(params.N_M,1);
else
    sigsq = sigStDev^2.*ones(params.N_M,1);
end

% g(:,1) = mean(abs(fx) + randn(size(fx))); % initialize at true value
% phi(:,1) = angle(fx) + .1*randn(size(fx));


AHfHat = AH(fHat);


if ~unitary
    switch forward_op
        case "blur"
            % Amatreal = sparse(real(Amat));
            % Amatimag = sparse(imag(Amat));
            % Amatreal = @(x) real(A(x));
            % Amatimag = @(x) imag(A(x));
        case "rand"
            Amatreal = @(x) real(A(x));
            Amatimag = @(x) imag(A(x));
            Amatfull = @(x) [real(A(x));imag(A(x))];
            AHmatfull = @(x) real(AH(x(1:end/2)))-imag(AH(x(end/2+1:end)));
    end
    

    % ARAR = Amatreal'*Amatreal;
    % AIAR = Amatimag'*Amatreal;
    % ARAI = Amatreal'*Amatimag;
    % AIAI = Amatimag'*Amatimag;
end

if strcmp(forward_op,"blur")
    sizeDepend = 5;
    if params.N2 == 1
        numIter = sizeDepend;
    else
        numIter = sizeDepend^2;
    end
    for ii = 1:numIter
        if params.N2 == 1
            currentSample = ii:sizeDepend:params.N1;
        else
            vIndex = ((floor((ii-1)/sizeDepend)+1):sizeDepend:params.N1)';
            hIndex = ((mod(ii-1,sizeDepend)+1):sizeDepend:params.N2);
            indexSetH = repmat(hIndex,length(vIndex),1);
            indexSetV = repmat(vIndex,length(hIndex),1);
            currentSample = sub2ind([params.N1,params.N2],indexSetH(:),indexSetV(:));    
        end
        sampleNumber = strcat('v',num2str(ii));
        samplesBlur.(sampleNumber) = currentSample;
    end
else
    samplesBlur = -1;
end

for kk = 1:params.N_M    
    if kk <= 10
        tic
    end


    % post_density(kk) = 1/sigStDev^2*sum(abs(fHat-A(g(:,kk).*exp(1i*phi(:,kk)))).^2) ...
    %                         + 1/2*sum(log(tausq(:,kk))) + 1/2*(L*g(:,kk))'*(1./tausq(:,kk).*(L*g(:,kk))) ...
    %                         + (size(L,1)+params.lambda_r-1)*log(etasq(kk)) ...
    %                         + params.lambda_delta/etasq(kk) + sum(tausq(:,kk)/etasq(kk)/2);
    % 
    % post_density(kk) = 1/sigStDev^2*sum(abs(fHat-A(g(:,kk).*exp(1i*phi(:,kk)))).^2) ...
    %                         + size(L,1)*1/2*log(etasq(:,kk)) + 1/sqrt(etasq(kk))*sum(abs(L*g(:,kk)));

    % if kk > 3
    %     if params.N2 == 1
    %         figure(123);subplot(1,3,1);
    %         figure(123);plot(g(:,kk));ylim([0.5 3.5]);title('Random Sample at Current Iteration')
    %         subplot(1,3,2);plot(post_density(3:kk));title('Value of -log Posterior')
    %         subplot(1,3,3);plot(1./etasq(3:kk));title('Value of eta to the -2')
    %         set(gcf,'Position',[100 100 1200 400]);
    %     end
    % 
    %     if params.N2 ~= 1
    %         figure(123);subplot(1,3,1);
    %         imagesc(reshape(g(:,kk),params.N1,params.N2));colorbar;clim([0.5 2.5]);title('Random Sample at Current Iteration')
    %         subplot(1,3,2);plot(post_density(2:kk));title('Value of -log Posterior')
    %         subplot(1,3,3);plot(etasq(2:kk));title('Value of etasq')
    %         set(gcf,'Position',[100 100 1200 400]);
    %     end
    % end
 
    if unitary
        % AHAPhi = speye(params.N1*params.N2);
        % Gamma = (2/sigStDev.^2*(AHAPhi) +...
        %     L.'*spdiags(tausq(:,kk).^(-1),0,size(L,1),size(L,1))*L);
        % cholGamma = chol(Gamma,'lower');
    else
        % AmatPhi = A(diag(exp(1i*phi(:,kk))));
        if strcmp(forward_op,'rand')
            % Dphi = diag(exp(1i*phi(:,kk)));
            AmatPhi = @(x) [real(A(exp(1i*phi(:,kk)).*x));imag(A(exp(1i*phi(:,kk)).*x))];
            AmatPhiH = @(x) real(exp(-1i*phi(:,kk)).*AH(x(1:end/2)))-imag(exp(-1i*phi(:,kk)).*AH(x(end/2+1:end)));
        elseif strcmp(forward_op,'blur')
            Dphi = spdiags(exp(1i*phi(:,kk)),0,params.N1*params.N2,params.N1*params.N2); %sparse(diag(exp(1i*phi(:,kk))));
            AmatPhi = [real(A(Dphi));imag(A(Dphi))];
        end
        % DphiR = real(Dphi);
        % DphiI = imag(Dphi);
        % AHAPhi = (DphiR * ARAR * DphiR - DphiI * AIAR * DphiR

        % AmatPhireal = Amatreal(real(Dphi)) - Amatimag(imag(Dphi));
        % AmatPhiimag = Amatreal(imag(Dphi)) + Amatimag(real(Dphi));
        % AHAPhi = AmatPhireal' * AmatPhireal + AmatPhiimag'*AmatPhiimag;
        % AHA = real(AmatPhi)'*real(AmatPhi) + imag(AmatPhi)'*imag(AmatPhi);
        % AHAPhi = exp(-1i*phi(:,kk)).*AH(A(Dphi));
    end
    
    % try chol(Gamma);
    % catch ME
    %     disp('Matrix is not symmetric positive definite')
    % end

    if unitary
        qvarphi = exp(-1i*phi(:,kk)).*AH(fHat);
        mu = abs(qvarphi).*cos(angle(qvarphi));
        AmatPhi = @(x) x;
    else
        % mu = AmatPhireal'*real(fHat) + AmatPhiimag'*imag(fHat);
    end

    % g(:,kk+1) = real(cholGammainv*randn(length(fHat),1) + 2/sigStDev.^2 * Gammainv * (abs(exp(-1i*phi(:,kk)).*AH(fHat)) .* cos(angle(exp(-1i*phi(:,kk)).*AH(fHat)))));
    % while nnz(g(:,kk+1)<0)>0
    %     g(:,kk+1) = real(cholGammainv*randn(length(fHat),1) + 2/sigStDev.^2 * Gammainv * (abs(exp(-1i*phi(:,kk)).*AH(fHat)) .* cos(angle(exp(-1i*phi(:,kk)).*AH(fHat)))));
    % end

    if kk < params.B-10
        adjust_funct = @(x) abs(x);
    else
        adjust_funct = @(x) x;
    end
    
    switch forward_op
        case "fft"
            % g(:,kk+1) = adjust_funct(cholGamma'\randn(params.N1*params.N2,1) + 2/sigStDev.^2 * (Gamma\mu));
            g(:,kk+1) = adjust_funct(perturbation_optimization_sampler(mu,tausq(:,kk),sqrt(sigsq(kk)),kk,params,AmatPhi,forward_op,L));
        case "blur"
            g(:,kk+1) = adjust_funct(perturbation_optimization_sampler([real(fHat);imag(fHat)],tausq(:,kk),sqrt(sigsq(kk)),kk,params,AmatPhi,forward_op,L));
        case "rand"
            g(:,kk+1) = adjust_funct(perturbation_optimization_sampler([real(fHat);imag(fHat)],tausq(:,kk),sqrt(sigsq(kk)),kk,params,AmatPhi,forward_op,L,AmatPhiH));
    end

    neg_count = 0;
    while nnz(g(:,kk+1)<0)>0
        switch forward_op
            case "fft"
                g(:,kk+1) = perturbation_optimization_sampler(mu,tausq(:,kk),sqrt(sigsq(kk)),kk,params,AmatPhi,forward_op,L);
            case "blur"
                g(:,kk+1) = perturbation_optimization_sampler([real(fHat);imag(fHat)],tausq(:,kk),sqrt(sigsq(kk)),kk,params,AmatPhi,forward_op,L);
            case "rand"
                g(:,kk+1) = perturbation_optimization_sampler([real(fHat);imag(fHat)],tausq(:,kk),sqrt(sigsq(kk)),kk,params,AmatPhi,forward_op,L,AmatPhiH);
        end
        neg_count = neg_count + 1;
        if neg_count > 10
            switch forward_op
                case "fft"
                    g(:,kk+1) = perturbation_optimization_sampler(mu,tausq(:,kk),sqrt(sigsq(kk)),kk,params,AmatPhi,forward_op,L);
                case "blur"
                    g(:,kk+1) = perturbation_optimization_sampler([real(fHat);imag(fHat)],tausq(:,kk),sqrt(sigsq(kk)),kk,params,AmatPhi,forward_op,L);
                case "rand"
                    g(:,kk+1) = perturbation_optimization_sampler([real(fHat);imag(fHat)],tausq(:,kk),sqrt(sigsq(kk)),kk,params,AmatPhi,forward_op,L,AmatPhiH);
            end
        end
        if neg_count > 2000
            error('Too negative, kk = %i',kk);
        end
    end
    neg_count = 0;

    if mod(kk,round(params.N_M/20)) == 0
        fprintf('At iteration %i\n',kk);
    end

    Lg = L*g(:,kk+1);

    if params.learn_sigma
        mu = sqrt(sigsq(kk)./(Lg.^2*etasq(kk)));
    else
        mu = sqrt(1./(Lg.^2*etasq(kk)));
    end
    lambda = 1/etasq(kk);
    v = randn(size(L,1),1);
    y = v.^2;
    x = mu + mu.^2.*y/(2*lambda) - mu/(2*lambda).*sqrt(4*mu*lambda.*y+mu.^2.*y.^2);
    test_unif = rand(size(L,1),1);
    tausq(:,kk+1) = 1./x .* (test_unif<=mu./(mu+x)) + 1./(mu.^2./x) .* (test_unif>mu./(mu+x));
    
    tausq(abs(Lg)<=1e-8,kk+1) = gamrnd(1/2,2*etasq(kk),nnz(abs(Lg)<=1e-8),1);

    % for jj = 1:size(L,1)
    %     mu = sqrt(1./(Lg(jj.^2*etasq(kk)));
    %     lambda = 1/etasq(kk);
    %     v = randn;
    %     y = v.^2;
    %     x = mu + mu.^2.*y/(2*lambda) - mu/(2*lambda).*sqrt(4*mu*lambda.*y+mu.^2.*y.^2);
    %     if rand <= mu./(mu+x)
    %         tausq(jj,kk+1) = 1./x;
    %     else
    %         tausq(jj,kk+1) = 1./(mu.^2./x);
    %     end
    %     if abs(Lg(jj)) <= 1e-10
    %         tausq(jj,kk+1) = gamrnd(1/2,2*etasq(kk));
    %     end
    % end


    if unitary
        phi(:,kk+1) = wrapCauchStepPhi(g(:,kk+1).*AHfHat,1./sqrt(sigsq(kk)),unitary,phi(:,kk));
    else
        if strcmp(forward_op,'blur')
            AmatG = A(spdiags(g(:,kk+1),0,params.N1*params.N2,params.N1*params.N2));
        else
            AmatG = A(diag(g(:,kk+1)));
        end
        % phi(:,kk+1) = wrapCauchStepPhi(g(:,kk+1).*AHfHat,1./sigStDev,unitary,phi(:,kk),AmatG'*AmatG,params);
        phi(:,kk+1) = wrapCauchStepPhiForwardOp(g(:,kk+1).*AHfHat,1./sqrt(sigsq(kk)),forward_op,phi(:,kk),AmatG'*AmatG,params,samplesBlur);
    end


    % AHA = real(AmatG)'*real(AmatG) + imag(AmatG)'*imag(AmatG);
    % mu = real(AmatG)'*real(fHat) + imag(AmatG)'*imag(fHat);

    % lambdasqsamp(kk+1) = gamrnd(size(L,1)+(params.lambda_r-1),1./(sum(tausq(:,kk+1)/2)+params.lambda_delta));
    % recip_eta_sq = gamrnd(size(L,1)+params.lambda_r,1./ (sum(tausq(:,kk+1)/2)+params.lambda_delta) );
    if and(params.N2~=1, strcmp(params.sparse_domain,"transform"))
        etasq(kk+1) = 0.01;
    else
        etasq(kk+1) = 1/gamrnd(size(L,1)+params.lambda_r,1/(sum(tausq(:,kk+1)/2)+params.lambda_delta)); %1/lambdasqsamp(kk+1);
    end

    if params.learn_sigma
        sigsq(kk+1) = 1/gamrnd(size(L,1)/2+params.N1*params.N2,1/(sum(abs(A(exp(1i*phi(:,kk+1)).*g(:,kk+1))-fHat).^2) + 1/2*sum(abs(1./sqrt(tausq(:,kk+1)).*Lg).^2)));
    end

    if kk <= 10
        toc
    end
end
g = g(:,params.B:end);
phi = phi(:,params.B:end);
tausq = tausq(:,params.B:end);
etasq = etasq(params.B:end);
end