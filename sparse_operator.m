function [L1,L2] = sparse_operator(params,separate,PAORDER,periodicBC)
% Inputs:
%   - params.N1 = dimension
%   - params.PAORDER = 0 if signal sparse, 1 if edges sparse
%
% Output:
%   - L = sparsifying transform
N1 = params.N1;
N2 = params.N2;

if params.N2 == 1
    switch PAORDER
        case {0}
            L1 = sparse(eye(N1));
        case {1}
            L1 = sparse(eye(N1) - circshift(eye(N1),-1,2));
            L1(1,:) = [1 zeros(1,N1-1)];
            % if ~periodicBC
            %     L1(1,:) = [];
            % end
    end
else
    switch PAORDER
        case {0}
            L1 = speye(N1 * N2);
            L2 = nan;
        case {1}
            if separate == 0
                T1 = 2*eye(N1) - circshift(eye(N1),-1,2);
                T2 = -eye(N1);
                L1 = sparse(N1*N2,N1*N2);
                L1(1:N1,1:N1) = T1;
                L1(1:N1,end-N1+1:end) = T2;
                for ii = 1:N2-1
                    L1(ii*N1+1:(ii+1)*N1,ii*N1+1:(ii+1)*N1) = T1;
                    L1(ii*N1+1:(ii+1)*N1,(ii-1)*N1+1:ii*N1) = T2;
                end
                L1 = sparse(L1);
            else
                T = eye(N1);
                T1 = -circshift(eye(N1),-1,2);
                T1(1,end) = 0;
                T2 = -eye(N1);
                L1 = sparse(N1*N2,N1*N2);
                L2 = sparse(N1*N2,N1*N2);
                L1(1:N1,1:N1) = T+T1;
                L2(1:N1,1:N1) = T;
                % L2(1:N1,end-N1+1:end) = T2;
                for ii = 1:N2-1
                    L1(ii*N1+1:(ii+1)*N1,ii*N1+1:(ii+1)*N1) = T+T1;
                    L2(ii*N1+1:(ii+1)*N1,ii*N1+1:(ii+1)*N1) = T;
                    L2(ii*N1+1:(ii+1)*N1,(ii-1)*N1+1:ii*N1) = T2;
                end
                L1 = sparse(L1);
                L2 = sparse(L2);
                % if ~periodicBC
                %     L1(1:N1:end,:) = [];
                %     L2(1:N2,:) = [];
                % end
            end
            L1 = [L1;L2];
            % L1 = [L1;L2(1:7000,:)];
    end     
end