function phiOut = wrapCauchStepPhi(xAHy,sigInv,unitary,phiIn,AHA,params)
% Sampling von Mises density using the wrapped Cauchy

phiOut = zeros(size(xAHy));
if unitary
    numIter = 1;
else
    numIter = length(phiOut);
end

for ii = 1:numIter
    if unitary
        mu = angle(xAHy);
        kappa = 2*sigInv^2*abs(xAHy);

        ongoing = logical(ones(size(phiOut)));
    else
        xAHyelse = xAHy; AHAelse = AHA(ii,:).';
        xAHyelse(ii) = []; AHAelse(ii) = [];
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
        if unitary
            phiOut(ongoing) = updateArray;
        else
            phiOut(ii) = updateArray;
        end
        ongoingLast = ongoing;
        ongoing(ongoing) = ~update;
    %     fprintf('%u\n',nnz(ongoing));
    end
end


end