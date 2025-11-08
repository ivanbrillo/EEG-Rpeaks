function X_pred_pulse = GPR_prediction(num_combinations,Y,combinations,gprModel_pulse)

    
    x_est_pulse=zeros(num_combinations,length(Y(1,:)));
    for ll=1:num_combinations
    
    indexes=combinations(ll,:);
    X_new = Non_Delay_Embedding(Y,indexes)'; % 10 new samples, 3 features

    [x_est_pulse(ll,:)] = predict(gprModel_pulse{ll}, X_new);
    end
   
   
    X_pred_pulse=mean(x_est_pulse,1);

end