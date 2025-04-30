function [U_final, Z_final,A_final, nIter_final, obj_all] = RNMF_Multi(X, Y, r, k, S, W, options, U, Z, A)
% Label Propagation Constrained NMF(LpCNMF)
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of dimensions 
%       nSmp  ... number of samples
% Y ... (nSmp x c)Label matrix of X
% r...  r << min(mFea , nSmp) number of hidden factors/subspace dimensions
% k ... number of classes
% S ... (nSmp x nSmp)diagonal label matrix
% W ... (nSmp x nSmp)weight matrix of the affinity graph 
% U ... (mFea x r) base
% Z ... (k x r) auxiliary
% A ... (nSmp x c) membership
% E ... (mFea x nFea) noise
% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = U*(AZ)'

% A = rand(nSmp,k);
differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat; %10
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;

alpha = options.alpha;

Norm = 2;
NormF = 1;

obj_all=[];

[mFea,nSmp]=size(X);
 
if alpha > 0
    W = alpha * W;
    S = alpha * S;
    
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,nSmp,nSmp);
    L = D - W;
    if isfield(options,'NormW') && options.NormW     
        D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;    
        L = D_mhalf * L * D_mhalf;
    end
else
    L = [];
end

selectInit = 1;
 % disp('Size of U:');
 % disp(size(U));
 % disp('U:');
 % disp(U);
if isempty(U)
    U = abs(rand(mFea, r));
    Z = abs(rand(k, r));
    A = abs(rand(nSmp, k));

    E = abs(rand(mFea, nSmp));%
else
    nRepeat = 1;  
end

[U, Z, A] = NormalizeUV(U, Z, A, NormF, Norm);   


if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, Z, A, S, Y, L);
        meanFit = objhistory*10;     
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, Z, A, S, Y, L);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end


tryNo = 0;
nIter = 0;
nCal=0;


Uf = opS(mFea,r); %????????????
U0 = U;
v = ones(r,1)./sqrt(r);
Lambda = rand(mFea, nSmp);

while tryNo < nRepeat   
    tryNo = tryNo+1;
    maxErr = 1;
    while maxErr > differror
        % ------------------------- begin algorithm -----------------------------
        lambda = 1000;
        beta = 10; %1000
        mu = 1;
        W = X - E - Lambda/beta;
        % disp('Size of E:');
        % disp(size(E));
        % disp('E:');
        % disp(E);
				
		% ===================== update U ========================
		Udelta = 10^4;
		Ugamma = 2;
		Ueta = 0.5 * (1/Ugamma);
		Ubeta = 0.5;
		Uepsg = 10^-1;
		Uepsmin = 10^-4;
		Ueps = 10^-4;
		max_outer_iters = 20;  % 限制外层迭代次数
		max_inner_iters = 5;   % 限制内层迭代次数
		max_armijo_iters = 5;  % 限制 Armijo 搜索的迭代次数

		Fu = @(U) 1/2 * norm(W - U * Z' * A','fro')^2 + Udelta * norm(U * v,'fro')^2;
		sFu = @(U) -(W - U * Z' * A') * A * Z + 2 * Udelta * U * v * v';

		for Uiter = 1:max_outer_iters
			if Fu(U0) > Fu(Uf)
				U0 = Uf;
			end

			for sta = 1:max_inner_iters
				dU = -sFu(U0);
				Fu0 = Fu(U0);
				Ualpha = 0.2;

				% Armijo search with limited iterations
				for v = 1:max_armijo_iters
					intU = U0 + Ualpha * dU;
					Uk = projOB(intU); % 投影

                    % [U, S1, V1] = svd(intU);
                    % Uk = U * S1 * V1';  % Reconstruct the matrix from SVD

					tempU = norm(U0 - Uk, 'fro')^2;
					Fuk = Fu(Uk);
					Ualpha = Ualpha * Ubeta;
					if Fuk <= Fu0 - Ualpha/2 * tempU
						break;
					end
				end

				gFu = sFu(Uk) - Uk * diag(diag(Uk' * sFu(Uk)));
				if norm(min(Uk, gFu), 'fro') <= Uepsg && Fuk <= Fu0 
					break;
				end
			end

			if norm(Uk * v, 'fro')^2 - 1 <= Ueps
				break;
			end

			U0 = Uk;
			Udelta = Ugamma * Udelta;
			Uepsg = max(Ueta * Uepsg, Uepsmin);
		end

		U = U0;
		% save ('U1.mat','U');


		% ===================== update A ========================   
		B = lambda * L + mu * S;
		C = -Z * Z';
		D = -beta * W' * U * Z' - mu * S * Y;
        
       % D_clean = D;
       % D_clean(isinf(D_clean) | isnan(D_clean)) = 0;
       % D = D_clean;
        
        barA = lyap(B, C, D);

		% 非负矩阵约束
		A = max(barA, 0);
       
       % ===================== update Z ========================
       barZ = pinv(A' * A) * A' * W' * U;
       Z = max(barZ,0);
        
       % ===================== update E ======================== 
       % V = X - U * Z * A' - Lambda/beta;
        %  tau = 3/2;
        % for i = 1 : size(E,1)
        %     Vi = norm(V(i,:));  % 使用欧几里得范数
        % 
        %     % Step 1: Apply L0 regularization
        %     if Vi <= sqrt(2/beta)
        %         Prox_L0 = 0;
        %     else
        %         Prox_L0 = Vi;  % Retain the original value for further processing
        %     end
        % 
        %     % Step 2: Apply MCP regularization
        %     if Prox_L0 == 0
        %         Prox_MCP = 0;
        %     elseif Prox_L0 >= lambda && Prox_L0 <= lambda * tau
        %         Prox_MCP = (tau * (Prox_L0 - lambda))/(tau - 1);
        %     else
        %         Prox_MCP = Prox_L0;
        %     end
        % 
        %     % Update E based on MCP regularization
        %     if Vi > 0
        %         E(i,:) = Prox_MCP * V(i,:)/Vi;
        %     else
        %         E(i,:) = 0;
        %     end
        % end
       V = X - U * Z * A' - Lambda/beta;
       for i = 1 : size(E,1)
           Vi = norm(V(i,:)); 
            % %phi = L_{0}
            % %change all vi into prox
            % if Vi <= sqrt(2/beta)
            %    Prox1 = 0;
            % else
            %    Prox1 = sqrt(2/beta);
            % end

           % %phi = L_{1/2}
           % Prox = 2/3 * Vi * (1 + cos(2 * pi/3 - 2/3 * acos(lambda/4 * (Vi/3)^(-3/2))));

           % %phi = capped L_{1} (tau > 1/2)
           % tau = 2/3;
           % if Vi <= lambda
           %     Prox = 0;
           % elseif Vi >= lambda & Vi <= lambda * (tau + 1/2)
           %     Prox = Vi - lambda;
           % elseif Vi == lambda * (tau + 1/2)
           %     Prox = lambda * tau - lambda/2;
           % else
           %     Prox = Vi;
           % end
           % 
           % % %% % phi = MCP (tau > 1)
           % tau = 3/2;
           % if Vi < lambda
           %     Prox = 0;
           % elseif Vi >= lambda & Vi <= lambda * tau
           %     Prox = (tau * (Vi - lambda))/(tau - 1);
           % else
           %     Prox = Vi;
           % end

           % % phi = SCAD (tau > 2)
           % % % change all vi into prox1
           % tau = 3.7;
           % if Vi <= lambda
           %     Prox = 0;
           % elseif Vi >= lambda & Vi <= 2 * lambda
           %     Prox = Vi - lambda;
           % elseif Vi >= 2 * lambda & Vi <= lambda * tau
           %     Prox = ((tau -1) * Vi - lambda * tau)/(tau - 2);
           % else
           %     Prox = Vi;
           % end 

%            % phi = Laplace
%            if abs(Vi) <= lambda
%                 Prox = 0;
%            else
%                 Prox = sign(Vi) * (abs(Vi) - lambda);
%            end

           % % % phi = ETP
           if Vi == 0
            Prox = 0;
           else
            Prox = sign(Vi) * (lambda / 2) * (sqrt(1 + 4 * abs(Vi) / lambda) - 1);
           end
           % tau = 2;
           % Prox = lambda / (1 - exp(-tau)) * (1 - exp(-tau * Vi));

           % % phi = Geman-McClure
           %  if Vi == 0
           %          Prox = 0;
           % else
           %          Prox = (2 * Vi) / (1 + sqrt(1 + 4 * lambda));
           % end

           % disp('Size of E:');
           % disp(size(E));

           % E(i,:) = (Prox * V(i,:)/norm(V(i,:)) + Prox1 * V(i,:)/norm(V(i,:)))/2;%Prox1 for symbol of L0
           % E(i,:) = Prox1 * V(i,:)/norm(V(i,:));
           E(i,:) = Prox * V(i,:)/norm(V(i,:));
           % E(i,:) = Prox * V(i,:)/norm(V(i,:)) + Prox1 * V(i,:)/norm(V(i,:));
       end
         % save ('SCADCOIL_E1.mat','E');

       
       % ==================== update Lambda ======================== 
       Lambda = Lambda - beta * (X - U * Z' * A' - E);
       % DA = U * Z' * A' + E;
       % DA = Normalize255(DA)/255; % -1~ 1 --> 0~255
       % DA = DA';
       % DAA = X;
       % DAAA = U * Z' * A';

      
         % save ('U2.mat','U');
      % ------------------------- end algorithm ----------------------------
        nIter = nIter + 1;
        if nCal < maxIter
            if nIter <= maxIter
                obj = CalculateObj(X, U, Z, A, S, Y, L);
                obj_all =[obj_all obj];

            end
            nCal = nCal + 1;
        end
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, Z, A, S, Y, L);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, Z, A, S, Y, L);
                    objhistory = [objhistory newobj];
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, Z, A, S, Y, L);
                        objhistory = [objhistory newobj];
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
                    end
                end
            end
        end
    end
    % save('data.mat', 'obj_all', 'objhistory');
    if tryNo == 1
        U_final = U;
        Z_final = Z;
        A_final = A;  
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           Z_final = Z;
           A_final = A;  
           nIter_final = nIter;
           objhistory_final = objhistory;
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(mFea, r));
            Z = abs(rand(k, r));
            A = abs(rand(nSmp, k));

            [U,Z,A] = NormalizeUV(U,Z,A,NormF,Norm);
        else
            tryNo = tryNo - 1;
            nIter = minIter+1;
            selectInit = 0;
            U = U_final;
            Z = Z_final;
            A = A_final;
            objhistory = objhistory_final;
            meanFit = objhistory * 10;
        end
    end
    % RU(tryNo,:,:) = U;
    % RZ(tryNo,:,:) = Z;
    % RA(tryNo,:,:) = A;
    % RE(tryNo,:,:) = E;
end

[U_final, Z_final, A_final] = NormalizeUV(U_final, Z_final, A_final, NormF, Norm);

%==========================================================================

function [obj, dV] = CalculateObj(X, U, Z, A, S, Y, L, deltaVU, dVordU)
    MAXARRAY = 500*1024*1024/8; % 500M. You can modify this number based on your machine's computational power.
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    nSmp = size(X,2);
    mn = numel(X);
    nBlock = ceil(mn/MAXARRAY);
    V = A * Z;

    if mn < MAXARRAY
        dX = X - U * V';
        obj_NMF = sum(sum(dX.^2));   %||X-UV'||2
        if deltaVU
            if dVordU
                dV = dX' * U + L * V;
            else
                dV = dX * V;
            end
        end
    else
        obj_NMF = 0;
        if deltaVU
            if dVordU
                dV = zeros(size(V));
            else
                dV = zeros(size(U));
            end
        end
        PatchSize = ceil(nSmp/nBlock);
        for i = 1:nBlock
            if i*PatchSize > nSmp
                smpIdx = (i-1)*PatchSize+1:nSmp;
            else
                smpIdx = (i-1)*PatchSize+1:i*PatchSize;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV(smpIdx,:) = dX'*U;
                else
                    dV = dU+dX*V(smpIdx,:);
                end
            end
        end
        if deltaVU
            if dVordU
                dV = dV + L*V;
            end
        end
    end
    if isempty(L)
        obj_Lap = 0;
    else
        sum1 = sum(sum((A' * L) .* A')); %tr(A'LA)+tr[(A-Y)'S(A-Y)]
        AY = A - Y; 
        sum2 = sum(sum(AY' * S .* AY'));
        obj_Lap = sum1 + sum2;      
    end
    obj = obj_NMF + obj_Lap;
    

function [U,Z,A] = NormalizeUV(U,Z,A,NormF,Norm)
    if Norm == 2         
        if NormF 
            norms = max(1e-15,sqrt(sum(A.^2,1)))'; 
            A = A * spdiags(norms.^-1,0,size(A,2),size(A,2));
            normsu = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U * spdiags(normsu.^-1,0,size(U,2),size(U,2));
            Z = Z;
            % Z = spdiags(sqrt(norms),0,size(A,2),size(A,2)) * Z * spdiags(sqrt(normsu),0,size(Z,2),size(Z,2));
        end
    else
        if NormF
            norms = max(1e-15,sum(abs(A),1))';
            A = A * spdiags(norms.^-1,0,size(A,2),size(A,2));
            U = U * spdiags(sqrt(norms),0,size(U,2),size(U,2));
            Z = Z * spdiags(sqrt(norms),0,size(Z,2),size(Z,2));
        end
    end
	
% 用共轭梯度法求解矩阵方程 AX + XB = D
function X = solve_sylvester(A, B, D)
    % 初始化
    [m, n] = size(D);
    X = zeros(m, n);
    max_iter = 1000;
    tol = 1e-6;

    % 初始化残差
    R = D - (A * X + X * B');
    P = R;

    for k = 1:max_iter
        % 计算 α
        AP = A * P + P * B';
        alpha = sum(sum(R .* R)) / sum(sum(P .* AP));
        
        % 更新 X
        X = X + alpha * P;
        
        % 更新残差
        R_new = R - alpha * AP;
        
        % 检查收敛性
        if norm(R_new, 'fro') < tol
            break;
        end
        
        % 更新 P
        beta = sum(sum(R_new .* R_new)) / sum(sum(R .* R));
        P = R_new + beta * P;
        
        % 更新残差
        R = R_new;
    end