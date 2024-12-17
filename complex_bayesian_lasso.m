clear all
close all

rng("default");

forward_operators = ["fft"];
SNRtotal = [20];

percent_zero_total = 0.2;%0:0.005:0.5; % only impacts "rand" forward operator

params.N1 = 200;
params.N2 = 1;
params.sparse_domain = "transform"; % signal or transform
params.test_image = "piper"; % transform case, "shepp" or "piper"
params.CCD = 0;
params.real = 0;

params.learn_sigma = 0;

createLegends = 0;

params.lambda_r = 1;%params.N1*params.N2;
params.lambda_delta = 0.001;%2*params.N1*params.N2*10^(-3);%0.01;

savefolder = strcat('figures/N',num2str(params.N1),'x',num2str(params.N2));
if ~exist(savefolder,'dir')
    mkdir(savefolder);
end

params.N_M = 5000;
params.B = 200;

% build ground truth
if params.N2 == 1
    physGrid = linspace(-pi,pi,params.N1);
else
    [xMesh,yMesh] = meshgrid(linspace(-pi,pi,params.N1),linspace(-pi,pi,params.N2));
    physGrid(:,:,1) = xMesh;
    physGrid(:,:,2) = yMesh;
end
fx = build_lasso_smv(params,physGrid);
fx = fx(:);

%% begin loop
for iiForOp = 1:length(forward_operators)
forward_op = forward_operators(iiForOp); % id or fft or blur or rand

if and(strcmp(params.sparse_domain,"transform"),params.N2==1)
    if length(percent_zero_total) >= 2
        avgPhaseDistFromTrue = zeros(length(percent_zero_total),1);
        mlePhaseDistFromTrue = zeros(length(percent_zero_total),1);
    else
        avgPhaseDistFromTrue = zeros(length(SNRtotal),1);
        mlePhaseDistFromTrue = zeros(length(SNRtotal),1);
    end
end

for iipercentZero = 1:length(percent_zero_total) % "parfor" if many percZero values
percent_zero = percent_zero_total(iipercentZero);

CVBL_error = zeros(length(SNRtotal),1);

for iiSNR = 1:length(SNRtotal) % "parfor" if many SNR values

SNR = SNRtotal(iiSNR);

% build forward model
switch forward_op
    case "id"
        A = @(x) x;
        AH = @(x) x;
        Ainv = @(x) AH(x);
        AHA = speye(params.N1*params.N2);
        Amat = speye(params.N1*params.N2);

        unitary = 1;
    case "fft"
        if params.N2 == 1
            A = @(x) 1/sqrt(params.N1)*fft(x);
            AH = @(x) sqrt(params.N1)*ifft(x);
        else
            A = @(x) 1/sqrt(params.N1*params.N2)*reshape(fft2(reshape(x,params.N1,params.N2,[])),params.N1*params.N2,[]);
            AH = @(x) sqrt(params.N1*params.N2)*reshape(ifft2(reshape(x,params.N1,params.N2,[])),params.N1*params.N2,[]);
        end
        Ainv = @(x) AH(x);
        AHA = speye(params.N1*params.N2);
        if params.N2 == 1
            Amat = A(eye(params.N1*params.N2));
        else
            Amat = [];
        end

        unitary = 1;
        periodicBC = 0;
    case "blur"
        if params.N2 == 1
            p = 1/sqrt(2+8+16)*[1 2 4 2 1].*ones(params.N1,5);
            Amat = spdiags(p,-2:2,params.N1,params.N1);
            % Amat = full(Amat);
            % [U,S,V] = svd(Amat);
            % Amat = V';
            A = @(x) Amat*x;
            AH = @(x) Amat'*x;
            Ainv = @(x) Amat\x;
            AHA = Amat'*Amat;

            unitary = 0;
        else
            % p = 1/sqrt(8+16+16)*[0 0 1 0 0;
            %                      0 1 2 1 0;
            %                      1 2 4 2 1;
            %                      0 1 2 1 0;
            %                      0 0 1 0 0];
            p = 1/sqrt(8+16+256)*[0 0 1 0 0;
                                  0 1 2 1 0;
                                  1 2 16 2 1;
                                  0 1 2 1 0;
                                  0 0 1 0 0];
            Amat = sparse(params.N1*params.N2,params.N1*params.N2);
            for jj = 1:params.N1*params.N2
                test_vec = zeros(params.N1,params.N2,1);
                test_vec(jj) = 1;
                Amat(:,jj) = sparse(reshape(conv2(test_vec,p,'same'),[],1));
            end
            A = @(x) Amat*x;
            AH = @(x) Amat'*x;
            Ainv = @(x) Amat\x;
            AHA = Amat'*Amat;

            unitary = 0;
        end
        periodicBC = 0;
    case "rand"
        undersampsize = ceil(params.N1*params.N2*percent_zero);
        undersample = ones(params.N1,params.N2);
        undersample(randperm(params.N1*params.N2-1,undersampsize)+1) = 0;
        undersample = logical(undersample);
        if params.N2 == 1
            A = @(x) 1/sqrt(params.N1)*delete_under(fft(x),params,undersample);
            AH = @(x) sqrt(params.N1)*ifft(resize_under(x,params,undersampsize,undersample));
        else
            A = @(x) 1/sqrt(params.N1*params.N2)*reshape(delete_under(fft2(reshape(x,params.N1,params.N2,[])),params,undersample),ceil(params.N1*params.N2-undersampsize),[]);
            AH = @(x) sqrt(params.N1*params.N2)*reshape(ifft2(resize_under(reshape(x,params.N1*params.N2-undersampsize,[]),params,undersampsize,undersample)),params.N1*params.N2,[]);
        end
        Ainv = @(x) AH(x);

        zero_vec = zeros(params.N1*params.N2,1);
        % Amat = A(eye(params.N1*params.N2));
        % Amat = zeros(round(params.N1*params.N2*(1-percent_zero)),params.N1*params.N2);
        % for jj = 1:params.N1*params.N2
        %     one_vec = zero_vec;one_vec(jj) = 1;
        %     Amat(:,jj) = A(one_vec);
        % end
        % AHA = Amat'*Amat;

        unitary = 0;
        periodicBC = 0;
end

%% compute data vector
% if strcmp(forward_op,"blur")
    % sigStDev = mean(abs(U*S*A(fx)))*10^(-SNR/20);
    % fHat = U*S*A(fx) + sigStDev/sqrt(2)*(randn(size(fx)) + 1i*randn(size(fx)));
    % fHat = S\U'*fHat;
    % sigStDev = sqrt(sigStDev^2*prod(diag(S))^(-2));
% else
    % sigStDev = sum(abs(A(fx)).^2)/length(A(fx))*10^(-SNR/20);
    fHat_true = A(fx);
    sigStDev = SNR_to_stdDev(fHat_true,SNR,params);
    fHat = fHat_true + sigStDev/sqrt(2)*(randn(length(fHat_true),1) + 1i*randn(length(fHat_true),1));
% end

%% Bayesian LASSO time

if strcmp(params.sparse_domain,"signal")
    PAORDER = 0;
    [a,b,tausq,etasq] = sparse_mag_lasso(fHat,A,AH,params,unitary,sigStDev,forward_op);
    saveVariablesSignal(savefolder,params,SNR,a,b,tausq,etasq,forward_op)
elseif strcmp(params.sparse_domain,"transform")
    PAORDER = 1;
    [g,phi,tausq,etasq] = sparse_transf_lasso(fHat,Ainv,A,AH,params,unitary,sigStDev,forward_op,periodicBC,fx);
    saveVariablesTransform(savefolder,params,SNR,g,phi,tausq,etasq,forward_op)
end

% if strcmp(forward_op,"blur")
%     if strcmp(params.sparse_domain,"signal")
%         a = V*a;
%         b = V*b;
%     elseif strcmp(params.sparse_domain,"transform")
%         g = V*g;
%         phi = V*phi;
%     end
% end

%% Plot results
close all
% if params.N2 == 256
%     clear tausq
% end
L = sparse_operator(params,1,PAORDER,periodicBC);
if params.N2 == 1
    figure(1);plot(physGrid,abs(fx),'k','LineWidth',1.5);hold on;
    figure(2);plot(physGrid,abs(fx),'k','LineWidth',1.5);hold on;
    if strcmp(params.sparse_domain,"signal")
        % Amat_phase = A(diag(exp(1i*angle(Ainv(fHat)))));
        % figure(1);plot(physGrid,lasso([real(Amat_phase);imag(Amat_phase)],[real(fHat);imag(fHat)],'Lambda',sigStDev^2/(2*params.N1*params.N2*mean(sqrt(etasq))),'Standardize',false),'r');
        if strcmp(forward_op,"rand")
            Amat = A(eye(params.N1));
        end
        z_lasso = complex_lasso(Amat,fHat,sigStDev^2/(2*mean(sqrt(etasq))),.1,1.5);
        figure(1);plot(physGrid,abs(z_lasso));
        figure(1);plot(physGrid,mean(sqrt(a.^2+b.^2),2),'b');hold off;
        xlim([-pi pi]);ylim([-0.25 1.75]);
    
        lowerquant = quantile(sqrt(a.^2+b.^2),0.05,2);
        upperquant = quantile(sqrt(a.^2+b.^2),0.95,2);
        figure(2);plot(physGrid,mean(sqrt(a.^2+b.^2),2),'b');
        shade(physGrid,lowerquant,'b',physGrid,upperquant,'b','FillType',[1 2;2 1]);hold off
        xlim([-pi pi]);ylim([-0.25 1.75]);
    
        positive_values = find(abs(fx)>0);
        figure(3);xline(angle(fx(positive_values(3))),'k--','LineWidth',1.5);hold on;
        invPhase = angle(z_lasso);xline(invPhase(positive_values(3)),'r-.','LineWidth',2);
        kde(angle(a(positive_values(3),:) + 1i*b(positive_values(3),:)));
        % lowerquant = quantile(angle(a(positive_values(3),:) + 1i*b(positive_values(3),:)),0.05,2);xline(lowerquant,'b:','LineWidth',1);
        % upperquant = quantile(angle(a(positive_values(3),:) + 1i*b(positive_values(3),:)),0.95,2);xline(upperquant,'b:','LineWidth',1);
        set(findall(gca, 'Type', 'Line'),'LineWidth',2);hold off;

        figure(4);plot(physGrid,real(fx),'k','LineWidth',1.5);hold on;
        plot(physGrid,mean(a,2),'r');hold on;
        lowerquant = quantile(a,0.05,2);
        upperquant = quantile(a,0.95,2);
        shade(physGrid,lowerquant,'r',physGrid,upperquant,'r','FillType',[1 2;2 1]);hold off
        xlim([-pi pi]);ylim([-1.75 1.75]);

        figure(5);plot(physGrid,imag(fx),'k','LineWidth',1.5);hold on;
        plot(physGrid,mean(b,2),'b');hold on;
        lowerquant = quantile(b,0.05,2);
        upperquant = quantile(b,0.95,2);
        shade(physGrid,lowerquant,'b',physGrid,upperquant,'b','FillType',[1 2;2 1]);hold off
        xlim([-pi pi]);ylim([-1.75 1.75]);
    
        CVBL_error(iiSNR) = 1/(params.N1*params.N2)*norm(fx-(mean(a,2)+1i*mean(b,2)),2)^2;
    elseif strcmp(params.sparse_domain,"transform")
        if strcmp(forward_op,"rand")
            Amat = A(eye(params.N1));
        end
        z_lasso = generalized_complex_lasso(Amat,fHat,L,sigStDev^2/(2*mean(sqrt(etasq))),.1);
        z_lasso_small = generalized_complex_lasso(Amat,fHat,L,sigStDev^2/(2*mean(sqrt(etasq))*10),.1);
        z_lasso_big = generalized_complex_lasso(Amat,fHat,L,sigStDev^2/(2*mean(sqrt(etasq))/10),.1);
        figure(1);plot(physGrid,abs(z_lasso),'r');
        figure(1);plot(physGrid,abs(z_lasso_big),'c');
        figure(1);plot(physGrid,abs(z_lasso_small),'m');
        figure(1);plot(physGrid,mean(g,2),'b');hold off;
        xlim([-pi pi]);ylim([0.5 2.75]);
    
        lowerquant = quantile(g,0.05,2);
        upperquant = quantile(g,0.95,2);
        figure(2);plot(physGrid,mean(g,2),'b');
        shade(physGrid,lowerquant,'b',physGrid,upperquant,'b','FillType',[1 2;2 1]);hold off
        xlim([-pi pi]);ylim([0.5 2.75]);
    
        figure(3);xline(angle(fx(60)),'k--','LineWidth',2);hold on;
        invPhase = angle(z_lasso);xline(invPhase(60),'r-.','LineWidth',2);
        kde(phi(60,:));
        % lowerquant = quantile(phi(60,:),0.05,2);xline(lowerquant,'b:','LineWidth',1);
        % upperquant = quantile(phi(60,:),0.95,2);xline(upperquant,'b:','LineWidth',1);
        set(findall(gca, 'Type', 'Line'),'LineWidth',2);hold off;

        figure(4);colororder({'k','b'});yyaxis right;
        plot(physGrid(1:size(L,1)),mean(tausq,2),'LineWidth',2);hold on;
        % lowerquant = quantile(tausq,0.25,2);
        % upperquant = quantile(tausq,0.75,2);
        % shade(physGrid,lowerquant,'b',physGrid,upperquant,'b','FillType',[1 2;2 1]);
        ylim([0 0.2]);yyaxis left;
        plot(physGrid,abs(fx),'--','LineWidth',2);hold off;
        ylim([0.5 2.75]);xlim([-pi pi]);
        
        CVBL_error(iiSNR) = 1/(params.N1*params.N2)*norm(fx-mean(g.*exp(1i*phi),2),2)^2;
     
        % MUST BE CHANGED DEPENDING ON WHAT PARFOR IS
        if length(percent_zero_total) >= 2
            % avgPhaseDistFromTrue(iipercentZero) = 1/(params.N1*params.N2)*norm(mean(phase_distance(abs(angle(fx)-phi)),2),2)^2;
            % mlePhaseDistFromTrue(iipercentZero) = 1/(params.N1*params.N2)*norm(phase_distance(abs(angle(fx)-invPhase)),2)^2;
        else
            avgPhaseDistFromTrue(iiSNR) = 1/(params.N1*params.N2)*norm(mean(phase_distance(abs(angle(fx)-phi)),2),2)^2;
            mlePhaseDistFromTrue(iiSNR) = 1/(params.N1*params.N2)*norm(phase_distance(abs(angle(fx)-invPhase)),2)^2;
        end
    end
else
    if strcmp(params.sparse_domain,"signal")
        figure(1);
        imagesc(xMesh(1:params.N1,1),yMesh(1:params.N2,1),reshape(mean(sqrt(a.^2+b.^2),2),params.N1,params.N2));
        % xlim([-pi pi]);ylim([-0.25 1.75]);
    
        % lowerquant = quantile(sqrt(a.^2+b.^2),0.05,2);
        % upperquant = quantile(sqrt(a.^2+b.^2),0.95,2);
        % figure(2);imagesc(xMesh(1:params.N1,1),yMesh(1:params.N2,1),reshape(upperquant-lowerquant,params.N1,params.N2));
        figure(2);imagesc(xMesh(1:params.N1,1),yMesh(1:params.N2,1),reshape(mean(tausq,2),params.N1,params.N2));
        % figure(2);plot(physGrid,mean(sqrt(a.^2+b.^2),2),'b');
        % shade(physGrid,lowerquant,'b',physGrid,upperquant,'b','FillType',[1 2;2 1]);hold off
        % xlim([-pi pi]);ylim([-0.25 1.75]);
        if params.CCD == 1
            [~,positive_values] = max(abs(fx));
        else
            positive_values = find(abs(fx)==1);
        end
        figure(3);xline(angle(fx(positive_values(1))),'k--','LineWidth',2);hold on;
        invPhase = angle(Ainv(fHat));xline(invPhase(positive_values(1)),'r--','LineWidth',2);
        kde(angle(a(positive_values(1),:) + 1i*b(positive_values(1),:)));
        % lowerquant = quantile(angle(a(positive_values(1),:) + 1i*b(positive_values(1),:)),0.05,2);xline(lowerquant,'b:','LineWidth',1);
        % upperquant = quantile(angle(a(positive_values(1),:) + 1i*b(positive_values(1),:)),0.95,2);xline(upperquant,'b:','LineWidth',1);
        set(findall(gca, 'Type', 'Line'),'LineWidth',2);hold off;
    
        CVBL_error(iiSNR) = 1/(params.N1*params.N2)*norm(fx-(mean(a,2)+1i*mean(b,2)),2)^2;
        
        figure(4);imagesc(reshape(abs(fx),params.N1,params.N2));colorbar;
    elseif strcmp(params.sparse_domain,"transform")  
        z_lasso = generalized_complex_lasso(Amat,fHat,L,sigStDev^2/(2*mean(sqrt(etasq))),.1,AH);

        figure(1);
        imagesc(xMesh(1:params.N1,1),yMesh(1:params.N2,1),reshape(mean(g,2),params.N1,params.N2));
        % xlim([-pi pi]);ylim([0.5 2.75]);
    
        lowerquant = quantile(g,0.05,2);
        upperquant = quantile(g,0.95,2);
        figure(2);imagesc(xMesh(1:params.N1,1),yMesh(1:params.N2,1),reshape(upperquant-lowerquant,params.N1,params.N2));
        % shade(physGrid,lowerquant,'b',physGrid,upperquant,'b','FillType',[1 2;2 1]);hold off
        % xlim([-pi pi]);ylim([0.5 2.75]);
    
        figure(3);xline(angle(fx(29550)),'k--','LineWidth',2);hold on;
        invPhase = angle(z_lasso);xline(invPhase(29550),'r-.','LineWidth',2);
        kde(phi(29550,:));
        set(findall(gca, 'Type', 'Line'),'LineWidth',2);hold off;
        
        CVBL_error(iiSNR) = 1/(params.N1*params.N2)*norm(fx-mean(g.*exp(1i*phi),2),2)^2;
    
        figure(4);imagesc(reshape(abs(fx),params.N1,params.N2));colorbar;

        figure(11);imagesc(reshape(abs(z_lasso),params.N1,params.N2));colorbar;
        figure(12);imagesc(reshape(abs(abs(z_lasso)-abs(fx)),params.N1,params.N2));colorbar;
        figure(13);imagesc(reshape(abs(mean(g,2)-abs(fx)),params.N1,params.N2));colorbar;
    end
end
% MLE_error = 1/(params.N1*params.N2)*norm(fx-AH(fHat),2)^2;

if params.N2 == 1
    figure(1);
    set(gcf,'Position',[100 100 600 300]);
    
    figure(2);
    set(gcf,'Position',[100 100 600 300]);
    
    figure(3);
    set(gcf,'Position',[100 100 500 300]);
    if strcmp(params.sparse_domain,"signal")
        figure(4);
        set(gcf,'Position',[100 100 600 300]);

        figure(5);
        set(gcf,'Position',[100 100 600 300]);
    end
else
    figure(1);colorbar;
    set(gcf,'Position',[100 100 600 400]);
    
    figure(2);colorbar;
    set(gcf,'Position',[100 100 600 400]);
    
    figure(3);
    set(gcf,'Position',[100 100 500 300]);

    figure(4);
    set(gcf,'Position',[100 100 600 400]);

    if strcmp(params.sparse_domain,"transform")
        figure(11);
        set(gcf,'Position',[100 100 600 400]);
    
        figure(12);
        set(gcf,'Position',[100 100 600 400]);
    
        figure(13);
        set(gcf,'Position',[100 100 600 400]);
    end
end

if strcmp(forward_op,"rand")
    savefile1 = strcat('Mean_',params.sparse_domain,'_',forward_op,'_percZero',num2str(100*percent_zero),'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
    savefile2 = strcat('CI_',params.sparse_domain,'_',forward_op,'_percZero',num2str(100*percent_zero),'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
    savefile3 = strcat('Phase_',params.sparse_domain,'_',forward_op,'_percZero',num2str(100*percent_zero),'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
else
    savefile1 = strcat('Mean_',params.sparse_domain,'_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
    savefile2 = strcat('CI_',params.sparse_domain,'_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
    savefile3 = strcat('Phase_',params.sparse_domain,'_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
end


if strcmp(params.sparse_domain,"signal")
    savefile4 = strcat('TrueMagSparseMag.png');
else
    savefile4 = strcat('TrueMagSparseEdge.png');
end

figure(1);saveas(gcf,strcat(savefolder,filesep,savefile1));
figure(2);saveas(gcf,strcat(savefolder,filesep,savefile2));
figure(3);saveas(gcf,strcat(savefolder,filesep,savefile3));

if params.N2 == 1
    if strcmp(forward_op,"rand")
        if strcmp(params.sparse_domain,"signal")
            savefile4 = strcat('real_',params.sparse_domain,'_',forward_op,'_percZero',num2str(100*percent_zero),'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
            figure(4);saveas(gcf,strcat(savefolder,filesep,savefile4));
    
            savefile5 = strcat('imag_',params.sparse_domain,'_',forward_op,'_percZero',num2str(100*percent_zero),'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
            figure(5);saveas(gcf,strcat(savefolder,filesep,savefile5));
        else
            savefile4 = strcat('tausq_',params.sparse_domain,'_',forward_op,'_percZero',num2str(100*percent_zero),'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
            figure(4);saveas(gcf,strcat(savefolder,filesep,savefile4));
        end
    else
        if strcmp(params.sparse_domain,"signal")
            savefile4 = strcat('real_',params.sparse_domain,'_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
            figure(4);saveas(gcf,strcat(savefolder,filesep,savefile4));
    
            savefile5 = strcat('imag_',params.sparse_domain,'_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
            figure(5);saveas(gcf,strcat(savefolder,filesep,savefile5));
        else
            savefile4 = strcat('tausq_',params.sparse_domain,'_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
            figure(4);saveas(gcf,strcat(savefolder,filesep,savefile4));
        end
    end
else
    figure(4);saveas(gcf,strcat(savefolder,filesep,savefile4));
    if strcmp(params.sparse_domain,"transform")
        savefile11 = strcat('genLASSO_',params.sparse_domain,'_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
        figure(11);saveas(gcf,strcat(savefolder,filesep,savefile11));

        savefile12 = strcat('genLASSODiff_',params.sparse_domain,'_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
        figure(12);saveas(gcf,strcat(savefolder,filesep,savefile12));

        savefile13 = strcat('CVBLDiff_',params.sparse_domain,'_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.png');
        figure(13);saveas(gcf,strcat(savefolder,filesep,savefile13));
    end
end

if createLegends == 1
    if params.N2 == 1
        figure(1);
        legend('True Magnitude','LASSO Magnitude $\alpha=1$','LASSO Magnitude $\alpha=10$','LASSO Magnitude $\alpha=0.1$',...
            'CVBL Magnitude Mean','Interpreter','latex');
        
        figure(2);
        legend('True Magnitude','CVBL Magnitude Mean','Interpreter','latex');
        
        figure(3);
        legend('True Phase','LASSO Phase','Est. Posterior of Phase','Interpreter','latex')
        if strcmp(params.sparse_domain,"signal")
            figure(4);
            legend('True Real','Mean Real','Interpreter','latex');

            figure(5);
            legend('True Imaginary','Mean Imaginary','Interpreter','latex')
        else
            figure(4);
            legend('True Magnitude','Mean of $\tau^2$','Interpreter','latex')
        end
    else
        figure(3);
        legend('True Phase','LASSO Phase','Est. Posterior of Phase')
    end

    if params.N2~=1
        figure(4);saveas(gcf,strcat(savefolder,filesep,savefile4));
        maxNumFig = 4;
    elseif strcmp(params.sparse_domain,"signal")
        maxNumFig = 5;
        legNames = ["Mean" "Mag" "Phase" "Real" "Imag"];
    else
        maxNumFig = 4;
        legNames = ["Mean" "Mag" "Phase" "Tausq"];
    end
    
    for jj = 1:maxNumFig
        figure(jj);
        % Call the legend to your choice, I used a horizontal legend here
        legend_handle = legend('Orientation','vertical');
        % Set the figure Position using the normalized legend Position vector
        % as a multiplier to the figure's current position in pixels
        % This sets the figure to have the same size as the legend
        set(gcf,'Position',(get(legend_handle,'Position')...
            .*[0, 0, 1, 1].*get(gcf,'Position')));
        % The legend is still offset so set its normalized position vector to
        % fill the figure
        set(legend_handle,'Position',[0,0,1,1]);
        % Put the figure back in the middle screen area
        set(gcf, 'Position', get(gcf,'Position') + [500, 400, 0, 0]);
        savelegend = strcat('legend',legNames(jj),'.png');
        saveas(gcf,strcat(savefolder,filesep,savelegend));
    end
end

end
end

if length(SNRtotal) >= 10
    if strcmp(forward_op,"rand")
        SNR_savefile = strcat('SNRcompare_',params.sparse_domain,'_',forward_op,'_percZero',num2str(100*percent_zero),'.mat');
    else
        SNR_savefile = strcat('SNRcompare_',params.sparse_domain,'_',forward_op,'.mat');
    end
    save(strcat(savefolder,filesep,SNR_savefile),"SNRtotal","CVBL_error");
    
    if and(strcmp(params.sparse_domain,"transform"),params.N2==1)
        distTrueSol_savefile = strcat('DistTrueSol',forward_op,'.mat');
        save(strcat(savefolder,filesep,distTrueSol_savefile),"SNRtotal",...
            "avgPhaseDistFromTrue","mlePhaseDistFromTrue");
    end
end

if length(percent_zero_total) >= 2
    underSampleDistTrueSol_savefile = strcat('underSampleDistTrueSol_SNR',num2str(SNRtotal(1)),'.mat');
    save(strcat(savefolder,filesep,underSampleDistTrueSol_savefile),"percent_zero_total",...
        "avgPhaseDistFromTrue","mlePhaseDistFromTrue");
end
end



function x = delete_under(x,params,undersample)
x = reshape(x,params.N1*params.N2,[]);
x(~undersample,:) = [];
end

function x_full = resize_under(x_under,params,undersampsize,undersample)
x_under = reshape(x_under,floor(params.N1*params.N2-undersampsize),[]);
num = size(x_under,2);
x_full = zeros(params.N1*params.N2,num);
x_full(undersample,:) = x_under;
x_full = reshape(x_full,params.N1,params.N2,[]);
end

function saveVariablesSignal(savefolder,params,SNR,a,b,tausq,etasq,forward_op)
save(strcat(savefolder,filesep,'Var_signal_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.mat'),"a","b","tausq","etasq","-v7.3");
end

function saveVariablesTransform(savefolder,params,SNR,g,phi,tausq,etasq,forward_op)
save(strcat(savefolder,filesep,'Var_transform_',forward_op,'_SNR',num2str(SNR),'_NM',num2str(params.N_M),'_B',num2str(params.B),'.mat'),"g","phi","tausq","etasq","-v7.3");
end

