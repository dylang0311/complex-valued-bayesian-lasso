function phiOut = wrapCauchStepPhiForwardOp(xAHy,sigInv,forward_op,phiIn,AHA,params,samples)
% Sampling von Mises density using the wrapped Cauchy

phiOut = zeros(size(xAHy));
switch forward_op
    case "fft"
        numIter = 1;
    case "blur"
        sizeDepend = 5;
        if params.N2 == 1
            numIter = sizeDepend;
        else
            numIter = sizeDepend^2;
            % AHA = reshape(params.N1,params.N2,params.N1*params.N2);
            % phiIn = reshape(phiIn,params.N1,params.N2);
        end
    case "rand"
        numIter = length(phiOut);
end

% if strcmp(forward_op,"blur")
%     for ii = 1:numIter
%         if params.N2 == 1
%             currentSample = ii:sizeDepend:params.N1;
%         else
%             vIndex = ((floor((ii-1)/sizeDepend)+1):sizeDepend:params.N1)';
%             hIndex = ((mod(ii-1,sizeDepend)+1):sizeDepend:params.N2);
%             indexSetH = repmat(hIndex,length(vIndex),1);
%             indexSetV = repmat(vIndex,length(hIndex),1);
%             currentSample = sub2ind([params.N1,params.N2],indexSetH(:),indexSetV(:));    
%         end
%         sampleNumber = strcat('v',num2str(ii));
%         samples.(sampleNumber) = currentSample;
%     end
% end

for ii = 1:numIter
    switch forward_op
        case "fft"
            mu = angle(xAHy);
            kappa = 2*sigInv^2*abs(xAHy);
    
            ongoing = logical(ones(size(phiOut)));
        case "blur" % THIS OPERATOR IS ASSUMED TO BE REAL
            sampleNumberii = strcat('v',num2str(ii));
            AHAelse = AHA(samples.(sampleNumberii),:).';
            AHAelse(samples.(sampleNumberii),:) = [];
            if params.N2 == 1
                if ii == 1
                    phiCurr = phiIn;
                    phiCurr(samples.(sampleNumberii)) = [];
                elseif ii == numIter
                    phiCurr = phiOut;
                    phiCurr(samples.(sampleNumberii)) = [];
                else
                    phiCurr = zeros(params.N1,1);
                    for jj = 1:sizeDepend
                        sampleNumberjj = strcat('v',num2str(jj));
                        if jj < ii
                            phiCurr(samples.(sampleNumberjj)) = phiOut(samples.(sampleNumberjj));
                        elseif jj > ii
                            phiCurr(samples.(sampleNumberjj)) = phiIn(samples.(sampleNumberjj));
                        end
                    end
                    phiCurr(samples.(sampleNumberii)) = [];
                end
            else
                if ii == 1
                    phiCurr = phiIn;
                    phiCurr(samples.(sampleNumberii)) = [];
                elseif ii == numIter
                    phiCurr = phiOut;
                    phiCurr(samples.(sampleNumberii)) = [];
                else
                    phiCurr = zeros(params.N1*params.N2,1);
                    for jj = 1:numIter
                        sampleNumberjj = strcat('v',num2str(jj));
                        if jj < ii
                            phiCurr(samples.(sampleNumberjj)) = phiOut(samples.(sampleNumberjj));
                        elseif jj > ii
                            phiCurr(samples.(sampleNumberjj)) = phiIn(samples.(sampleNumberjj));
                        end
                    end
                    phiCurr(samples.(sampleNumberii)) = [];
                end
            end
            u = 2*abs(xAHy(samples.(sampleNumberii)))*sigInv^2.*cos(angle(xAHy(samples.(sampleNumberii)))) - 2*sum(abs(AHAelse)*sigInv^2.*cos(phiCurr),1).';
            v = 2*abs(xAHy(samples.(sampleNumberii)))*sigInv^2.*sin(angle(xAHy(samples.(sampleNumberii)))) - 2*sum(abs(AHAelse)*sigInv^2.*sin(phiCurr),1).';
            
            mu = -atan(-v./u).*(u>0) + -(atan(-v./u)+pi).*(u<0) + -pi/2.*(u==0);
            kappa = sqrt(u.^2+v.^2);
    
            ongoing = logical(ones(size(samples.(sampleNumberii))));

        case "rand"
            AHAelse = AHA(ii,:).';
            AHAelse(ii) = [];
            if ii == 1
                phiCurr = phiIn(2:end);% + angle(AHAelse);
            elseif ii == numIter
                phiCurr = phiOut(1:end-1);% - angle(AHAelse);
            else
                phiCurr = [phiOut(1:ii-1);phiIn(ii+1:end)];% + [-angle(AHAelse(1:ii-1));angle(AHAelse(ii:end))];
            end
            phiCurr = phiCurr + angle(AHAelse);
            u = 2*abs(xAHy(ii))*sigInv^2*cos(angle(xAHy(ii))) - 2*abs(AHAelse).'*sigInv^2*cos(phiCurr);
            v = 2*abs(xAHy(ii))*sigInv^2*sin(angle(xAHy(ii))) - 2*abs(AHAelse).'*sigInv^2*sin(phiCurr);
            if u > 0
                mu = -atan(-v/u);
            elseif u < 0
                mu = -(atan(-v/u) + pi);
            else
                mu = -pi/2;
            end
            kappa = sqrt(u^2+v^2);
    
            ongoing = 1;
    end
    
    % Step 1
    tau = 1+sqrt(1+4*kappa.^2);
    rho = (tau-sqrt(2*tau))./(2*kappa);
    r = (1+rho.^2)./(2*rho);
    while sum(ongoing) > 0
        muOn = mu(ongoing);
        
        % Step 2
        u1 = rand(sum(ongoing),1);
        z = cos(pi*u1);
        f = (1+r(ongoing).*z)./(r(ongoing)+z);
        c = kappa(ongoing).*(r(ongoing)-f);
        
        % Step 3
        u2 = rand(sum(ongoing),1);
        update = (c.*(2-c) - u2) > 0;
        
        % Step 4
        update(~update) = (log(c(~update)./u2(~update)) + 1 - c(~update)) > 0;
        
        % Step 5
        u3 = rand(sum(update),1);
        updateArray = zeros(size(update));
        updateArray(update) = mod(sign(u3 - 0.5) .* acos(f(update)) + muOn(update) + pi,2*pi) - pi;
        switch forward_op
            case "fft"
                phiOut(ongoing) = updateArray;
            case "blur"
                currentSample = samples.(sampleNumberii);
                phiOut(currentSample(ongoing)) = updateArray;
            case "rand"
                phiOut(ii) = updateArray;
        end
        ongoingLast = ongoing;
        ongoing(ongoing) = ~update;
    %     fprintf('%u\n',nnz(ongoing));
    end
end


end