
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