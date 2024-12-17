function x = generalized_complex_lasso(F, y, L, lambda, rho, AH)

MAX_ITER = 50;
[k, n] = size(L);

x = zeros(n,1);
z = zeros(k,1);
u = zeros(k,1);
LTheta = speye(k,n);

for ii = 1:MAX_ITER
    if isempty(F) % only true for 2D FFT forward op
        x = (speye(n) + rho*(LTheta'*LTheta))\(AH(y) + rho*LTheta'*(z-u));
    else
        x = (F'*F + rho*(LTheta'*LTheta))\(F'*y + rho*LTheta'*(z-u));
    end
    Theta = x./abs(x).*speye(length(x));
    LTheta = L*Theta';
    z = wthresh(real(L*Theta'*x+u),"s",lambda/rho);
    u = real(u + L*Theta'*x - z);
end

end