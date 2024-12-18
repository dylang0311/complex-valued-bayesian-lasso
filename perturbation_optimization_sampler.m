function x_s = perturbation_optimization_sampler(y,tausq,sigStandDev,kk,params,H,forward_op,G,HH)

if params.learn_sigma % currently ignores all other "params.learn_sigma" instances
    Gamma = G'*(diag(1./tausq)*G) + 2*eye(size(G));
    Mean = Gamma\(2*y);
    x_s = mvnrnd(Mean,sigStandDev^2*inv(Gamma));
    return
end

switch forward_op
    case "fft"
        Hy = y;
    case "blur"
        Hy = H'*y;
    case "rand"
        Hy = HH(y);
end

if G == 1
    switch forward_op
        case "fft"
            Hy_tilde = Hy + sigStandDev/sqrt(2).*randn(length(y),1);
        case "blur"
            Hy_tilde = Hy + sigStandDev/sqrt(2).*(H'*randn(length(y),1));
        case "rand"
            Hy_tilde = Hy + sigStandDev/sqrt(2).*HH(randn(length(y),1));
    end
    mu_tilde = sqrt(tausq).*randn(size(tausq,1),1);  %mvnrnd(mu_p,sigma_p).';
    if strcmp(forward_op,"blur")
        Aconjgrad = sparse(2/sigStandDev^2*(H'*H) + diag(1./tausq));
        alpha = max(sum(abs(Aconjgrad),2)./diag(Aconjgrad))-2;
        L = ichol(Aconjgrad,struct('type','ict','droptol',1e-3,'diagcomp',alpha));
        [x_s,fl0,rr0,it0,rv0] = pcg(Aconjgrad,(1./tausq).*mu_tilde + 2/sigStandDev^2*Hy_tilde,1e-6,200,L,L');
    elseif strcmp(forward_op,"rand")
        Aconjgrad = @(x) 2/sigStandDev^2*HH(H(x)) + (1./tausq).*x;
        [x_s,fl0,rr0,it0,rv0] = pcg(Aconjgrad,(1./tausq).*mu_tilde + 2/sigStandDev^2*Hy_tilde,1e-6,200);
    end
else
    switch forward_op
        case "fft"
            if params.learn_sigma
                Aconjgrad = 2/sigStandDev^2*speye(size(G,2)) + 1/sigStandDev*G'*((1./tausq).*G);
            else
                Aconjgrad = 2/sigStandDev^2*speye(size(G,2)) + G'*((1./tausq).*G);
            end
        case "blur"
            Aconjgrad = sparse(2/sigStandDev^2*(H'*H) + G'*((1./tausq).*G));
        case "rand"
            Aconjgrad = @(x) 2/sigStandDev^2*HH(H(x)) + (G'*((1./tausq).*G))*x;
    end
    success = 0;
    iteration = 0;
    while ~success
        if or(kk<=params.B,iteration<10)
            RSM = 0;
            switch forward_op
                case "fft"
                    Hy_tilde = Hy + sigStandDev/sqrt(2).*randn(length(y),1);
                case "blur"
                    Hy_tilde = Hy + sigStandDev/sqrt(2).*(H'*randn(length(y),1));
                case "rand"
                    Hy_tilde = Hy + sigStandDev/sqrt(2).*HH(randn(length(y),1));
            end
    
            if params.learn_sigma
                mu_tilde = sigStandDev*sqrt(tausq).*randn(size(tausq,1),1);  %mvnrnd(mu_p,sigma_p).';
            else
                mu_tilde = sqrt(tausq).*randn(size(tausq,1),1);
            end
        
            if strcmp(forward_op,"fft")
                L = ichol(Aconjgrad,struct('type','ict','droptol',1e-3));
                if params.learn_sigma
                    [x_s,fl0,rr0,it0,rv0] = pcg(Aconjgrad,G'*((1./(tausq*sigStandDev^2)).*mu_tilde) + 2/sigStandDev^2*Hy_tilde,1e-6,200,L,L');
                else
                    [x_s,fl0,rr0,it0,rv0] = pcg(Aconjgrad,G'*((1./tausq).*mu_tilde) + 2/sigStandDev^2*Hy_tilde,1e-6,200,L,L');
                end
            elseif strcmp(forward_op,"blur")
                alpha = max(sum(abs(Aconjgrad),2)./diag(Aconjgrad))-2;
                L = ichol(Aconjgrad,struct('type','ict','droptol',1e-3,'diagcomp',alpha));
                [x_s,fl0,rr0,it0,rv0] = pcg(Aconjgrad,G'*((1./tausq).*mu_tilde) + 2/sigStandDev^2*Hy_tilde,1e-6,200,L,L');
            elseif strcmp(forward_op,"rand")
                [x_s,fl0,rr0,it0,rv0] = pcg(Aconjgrad,G'*((1./tausq).*mu_tilde) + 2/sigStandDev^2*Hy_tilde,1e-6,200);
            end
        else
            if iteration == 10
                if strcmp(forward_op,"rand")
                    Hmat = H(eye(params.N1*params.N2));
                    Aconjgrad = sparse(2/sigStandDev^2*(Hmat'*Hmat) + G'*((1./tausq).*G));
                    mode_orig = Aconjgrad\(2/sigStandDev^2*Hmat'*y);
                else
                    mode_orig = Aconjgrad\(2/sigStandDev^2*H'*y);
                end
                Achol = chol(Aconjgrad,'lower');
                
                x = optimvar('x',params.N1*params.N2);
                qprob = optimproblem;
                obj = 1/2*x'*Aconjgrad*x - mode_orig'*Aconjgrad*x;
                qprob.Objective = obj;
                cons = x >= 0;
                qprob.Constraints = cons;
                options = optimoptions('quadprog','Algorithm','interior-point-convex',...
                    'LinearSolver','sparse','StepTolerance',0);
                mode = solve(qprob,'Options',options);
                mode = mode.x;

                RSM = 1;
            end
            x_s = mode + Achol\randn(size(mode));
        end

        if or(sum(x_s<0)==0, strcmp(params.sparse_domain,"signal"))
            if RSM == 1
                u = rand;
                RSM_test_value = mode.'*(Aconjgrad*mode) - x_s.'*(Aconjgrad*mode);
                if log(u)<=RSM_test_value
                    success = 1;
                    fprintf('Successful at iteration %i\n',iteration);
                end
            else
                success = 1;
            end
        end

        if kk<=params.B
            success = 1;
        end

        iteration = iteration + 1;
    end
end
end
