function fx = build_lasso_smv(params,physGrid)
fx = zeros(params.N1,params.N2);
switch params.sparse_domain
    case "signal"
        if params.CCD == 1
            load('/Users/dylangreen/Desktop/ATRC_Laptop/Emp_Bayes_Complex_code/BayesianLASSO/CCD-CP-CoPol/HH/FP0120/c00007a283p50.mat')
            fx = SARdataOut(2300:2299+params.N1,2900:2899+params.N2);
        else
            if params.N2 == 1
                fx(randi(params.N1*params.N2,round(params.N1*params.N2/20),1)) = 1;
            else
                fx(randi(params.N1*params.N2,25,1)) = 1;
            end
            fx = fx.*(randi(5,size(fx,1),size(fx,2))./5+0.2);
        end
    case "transform"
        if params.N2 == 1
            fx = (physGrid>0)*1.5.*exp(-((physGrid-pi/2)*1.5).^2);
            fx(and(physGrid>-2.8,physGrid<-2.1)) = 1;
            fx(and(physGrid>-1.6,physGrid<-1.3)) = 0.5;
            fx = fx + 1;
        elseif params.N2 ~= params.N1
            fx = fx + 0.1;
            fx(sqrt(physGrid(:,:,1)'.^2+physGrid(:,:,2)'.^2)<2) = 1;
            fx(and(abs(physGrid(:,:,1)-1)'<1,abs(physGrid(:,:,2)-1)'<1)) = 0.5;
            fx = fx + 1;
        else
            if strcmp(params.test_image,"shepp")
                fx = 1+phantom(params.N1);
            elseif strcmp(params.test_image,"piper")
                fx = imresize(double(rgb2gray(imread("piper_test_image.jpeg"))),[params.N1,params.N2]);
                fx = fx./max(fx,[],'all') + 1;
            end
        end
end

if params.real == 0
    fx = fx .* exp(1i * 2*pi * (rand(size(fx))));
end
end