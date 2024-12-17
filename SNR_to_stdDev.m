function sigma = SNR_to_stdDev(fHatTrue,SNR,p)
sigma = sqrt(10^(-SNR/10)*norm(fHatTrue,2).^2/(p.N1*p.N2));

%sigma_test = sqrt(10^(-SNR/10))*norm(fHatTrue,2).^2/(p.N1*p.N2); % incorrect
end