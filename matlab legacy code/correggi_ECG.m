function ECG_corr = correggi_ECG(ecg_pulse_pred,threshold)
        ECG_corr=ecg_pulse_pred;
        pos = []; 
        ii=2;
    while ii<=length(ecg_pulse_pred)-1
        if(ecg_pulse_pred(ii)-ecg_pulse_pred(ii-1)<threshold)
           pos(end+1) = ii;
           ii=ii+2;
        else
            ii=ii+1;
        end
    end
    ECG_corr(pos)=[];

end



function [signal]=train_pulse(X,T,A,width,freq,varargin)
% Parameters
t = 0:1/freq:(length(X)-1)/freq;            % Time vector for evaluation
T=t(T);
% Initialize the signal
signal = zeros(size(t));

if(varargin{1}=='Rectangular')
% Generate the rectangular pulse train
for i = 1:length(T)
    % Logical indexing to find the pulse region
    pulse_region = (t >= T(i) - width/2) & (t <= T(i) + width/2);
    signal(pulse_region) = A;  % Set amplitude for the pulse
end
elseif(varargin{1}=='Gaussian')
    for i = 1:length(T)
    % Add Gaussian pulse centered at T(i)
    signal = signal + A * exp(-(t - T(i)).^2 / (2 * width^2));
    end
end
end



function [X_nemb]=Non_Delay_Embedding(X,l)
    % X is the original multivariate series, row vector
    % l is the embedding dimension vector index
    X_nemb=X(l,:);
end



function X_pred_pulse = GPR_prediction(num_combinations,Y,combinations,gprModel_pulse)

    
    x_est_pulse=zeros(num_combinations,length(Y(1,:)));
    for ll=1:num_combinations
    
    indexes=combinations(ll,:);
    X_new = Non_Delay_Embedding(Y,indexes)'; % 10 new samples, 3 features

    [x_est_pulse(ll,:)] = predict(gprModel_pulse{ll}, X_new);
    end
   
   
    X_pred_pulse=mean(x_est_pulse,1);

end




function gprModel_pulse = GPR_model(num_combinations,combinations,X,ECG_pulse_delay_cat)
    gprModel_pulse={zeros(num_combinations,1)};
 for ll=1:num_combinations
        
        indexes=combinations(ll,:);
        
        X_nemb=Non_Delay_Embedding(X,indexes);
         
        
        tic;

gprModel_pulse{ll,1} = fitrgp(X_nemb', ECG_pulse_delay_cat','KernelFunction', 'matern32','FitMethod','fic', ...
                                'PredictMethod','fic', ...
                                'ActiveSetSize',150, ...
                                'ActiveSetMethod', 'random','OptimizeHyperparameters', 'auto','HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus','ShowPlots', false,'Verbose',0,UseParallel=true));


        toc;
    end
end



function [X_emb]=Delay_Embedding(X,m)
    % X is the original monovariate series
    % m is the embedding dimension
    X=reshape(X,1,[]); 
    N=length(X);
    for ii=1:m
    X_emb(ii,:)=X(ii:N-m+ii);
    end
end



function combinazioni = combinazioni_random(N,k,n_combinazioni)
    if(k>N)
        k=N;
    end
    for nn=1:n_combinazioni
        combinazioni(nn,:) = randperm(N,k);
    end
end