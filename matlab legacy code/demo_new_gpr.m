%% Parametri per le simulazioni
clc;
clear all;
close all;
% Amplitude of the train pulses
A = 2;       
% Width of each pulse
width = 0.05;   
% Lag 
tau=5;             
% embedding dimension
L=10;           

 % numero di combinazioni che impiego per la previsione
n_combinations=10;

% riproducibilità
rng(42);            

% definisco le combinazioni su 90 canali
combinations = combinazioni_random(90,L,n_combinations); 

% Tempo di allenamento in secondi
time_training=10; 

% Tempo di test in secondi
time_testing=10;

%soggetti disponibili 
subj_all=[23,27,28,29,30];

% impulso per trasformare le serie ecg
impulso="Gaussian";


%% Costruisco le serie per l'addestramento
X_prepro=[];

ECG_cat=[];

ECG_pulse_delay_cat_prepro=[];

% definisco quali soggetti scelgo per l'addestramento tra subject all
subj_training = [27];

% definisco quali soggetti scelgo per la predizione
subj_test= 23;

for ss=subj_training

      % Carico le serie preprocessate, ovvero matrici d x N, dove
      % d è il numero delle features o dei canali
      % N è il numero degli istanti temporali

      load(sprintf('Signals/P0%02d_prepro.mat',ss),"EEG_prepro");
      Y_prepro=double(EEG_prepro.data);
        
      %definisco le frequenza di campionamento
      freq_prepro=EEG_prepro.srate;

      %taglio le serie e il vettore tempo al tempo di addestramento
      Y_prepro=Y_prepro(:,1:time_training*freq_prepro);
      t_prepro=[1/freq_prepro:1/freq_prepro:time_training]; % total time in seconds
        


      % carico le serie ECG, troncandole al tempo di addestramento
      load(sprintf('Signals/P0%02d.mat',ss));
      % R_peak serie dei picchi R
      R_peak(R_peak>size(Y_prepro,2))=[];
      % ECG 
      ECG = ECG_i;
      ECG(length(t_prepro):end) = []; % target
      ECG=ECG-mean(ECG);
      % Invece delle serie ECG, ho un treno di impulsi Gaussiani 
      % nei picchi R
      [ECG_pulse]=train_pulse(Y_prepro,R_peak,A,width,freq_prepro,impulso);


      % Mostro il grafico delle serie ECG e del treno di impulsi Gaussiano
       figure
      plot(t_prepro(1:size(ECG,2)),ECG)
      hold on
      plot(t_prepro(1:size(ECG_pulse,2)),ECG_pulse)
      title(sprintf('ECG e Impulsi Concatenati, Soggetto %d',ss))
      legend("ECG","Impulsi")
      xlabel("Time")
      drawnow
      hold off;
      
      
      % normalizzo i dati EEG e ECG 
     

      Y_prepro=normalize(Y_prepro,2,"range",[0,1]);
      Y_prepro=normalize(Y_prepro,2,"center","mean");    
      ECG_pulse=normalize(ECG_pulse,2,"range",[0,1]);
        
      
      
        % downsampling
        t_ds=t_prepro(1:tau:end);
        freq_prepro=floor(freq_prepro/tau);
        Y_prepro=Y_prepro(:,1:tau:end);
        ECG_pulse_delay_prepro=ECG_pulse(1:tau:end);
        ECG=ECG(1:tau:end);

        % concatenazione tra le serie dei vari soggetti
        X_prepro=[X_prepro,Y_prepro];
        ECG_pulse_delay_cat_prepro=[ECG_pulse_delay_cat_prepro,ECG_pulse_delay_prepro];
        ECG_cat=[ECG_cat,ECG];
        
end


%% Mostro la serie degli ECG concatenata insieme agli impulsi
 figure;
 plot(1/freq_prepro:1/freq_prepro:length(ECG_cat)*1/freq_prepro,ECG_cat)
 hold on
 plot(1/freq_prepro:1/freq_prepro:length(ECG_pulse_delay_cat_prepro)*1/freq_prepro,ECG_pulse_delay_cat_prepro)
 title('ECG e Impulsi Concatenati, downsampled')
xlabel("Time")
legend("ECG","Impulsi")
hold off;
drawnow




%% Fase di Addestramento
% Addestro il modello usando n_combinations tra quelle create.
% Per ogni combinazione trovo il predittore mediante una regressione con GP
gprModel_pulse_prepro = GPR_model(n_combinations,combinations,X_prepro,ECG_pulse_delay_cat_prepro);


%% Test su altri soggetti
% Previsione
X_pulse_pred_prepro = [];

ss=subj_test;

    % Carico i dati EEG, definendo frequenze e tagliando al tempo di
    % training
    load(sprintf('Signals/P0%02d_prepro.mat',ss),"EEG_prepro");
    
    Y_prepro=double(EEG_prepro.data);

    freq_prepro=EEG_prepro.srate;

    Y_prepro=Y_prepro(:,1:time_testing*freq_prepro);

    % Normalizzo i dati EEG preprocessati


    Y_prepro=normalize(Y_prepro,2,"range",[0,1]);
    Y_prepro=normalize(Y_prepro,2,"center","mean");  


    % Carico la serie ECG per confronto
    load(sprintf('Signals/P0%02d.mat',ss));
    R_peak(R_peak>size(Y_prepro,2))=[];

    T=R_peak;
    [ECG_target_pulse]=train_pulse(Y_prepro,T,A,width,freq_prepro,impulso);
    ECG_target_pulse=normalize(ECG_target_pulse,2,"range",[0,1]);

    t_prepro=[0:1/freq_prepro:time_testing];
    ECG_target = ECG_i;
    ECG_target(length(t_prepro)+1:end) = []; % target
    ECG_target=ECG_target-mean(ECG_target);

    % downsampling
    freq_prepro=floor(freq_prepro/tau);
    Y_prepro=Y_prepro(:,1:tau:end);
    ECG_target_pulse=ECG_target_pulse(1:tau:end);
    ECG_target=ECG_target(1:tau:end);
    t_ds=t_prepro(1:tau:end);


%% Predizione effettiva
X_pred_pulse_prepro = GPR_prediction(n_combinations,Y_prepro,combinations,gprModel_pulse_prepro);

    X_pred_pulse_interp_prepro=interp1(t_ds(1:length(X_pred_pulse_prepro)),X_pred_pulse_prepro,t_prepro,'cubic');
    X_pred_pulse_smooth_prepro=smoothdata(X_pred_pulse_interp_prepro,'loess',500*1);    

    freq_prepro=500;
    
    time_delay=500*0.5;
    [~,pos_pulse_pred_prepro]=findpeaks(X_pred_pulse_smooth_prepro,'MinPeakDistance',time_delay);
    time_ecg=t_prepro(R_peak);

    ecg_pulse_pred_prepro=t_prepro(pos_pulse_pred_prepro);
    ecg_pulse_pred_prepro=correggi_ECG(ecg_pulse_pred_prepro,0.6);

    distances_pulse_prepro=abs(ecg_pulse_pred_prepro'-time_ecg);
    [minDistances_pulse_prepro, indices] = min(distances_pulse_prepro, [], 2);

    RMSE_prepro=sqrt(mean(minDistances_pulse_prepro.^2));

    % Picchi Predetti
    picchi_predetti_prepro=length(ecg_pulse_pred_prepro);

    % Picchi effettivi
    picchi_effettivi_prepro=length(time_ecg);

    % Percentuale
    picchi_percentuale_prepro=sprintf("%.2f",length(ecg_pulse_pred_prepro)/length(time_ecg)*100);

    stringa = "Picchi predetti : "+num2str(length(ecg_pulse_pred_prepro))+" su "+num2str(length(time_ecg)) + " ("+sprintf("%.2f",length(ecg_pulse_pred_prepro)/length(time_ecg)*100)+"%)";
    disp(stringa)

    stringa = "Picchi <0.05 : "+num2str(sum(minDistances_pulse_prepro<0.05))+" su "+num2str(length(time_ecg)) + " ("+sprintf("%.2f",sum(minDistances_pulse_prepro<0.05)/length(time_ecg)*100)+"%)";
    disp(stringa)

     stringa = "Picchi <0.1 : "+num2str(sum(minDistances_pulse_prepro<0.1))+" su "+num2str(length(time_ecg)) + " ("+sprintf("%.2f",sum(minDistances_pulse_prepro<0.1)/length(time_ecg)*100)+"%)";
    disp(stringa)


    disp(append("RMSE ",num2str(RMSE_prepro)))

%% Plots
figure
    
plot(t_ds(1:length(ECG_target)),ECG_target)
hold on;
plot(t_ds(1:length(ECG_target_pulse)),ECG_target_pulse)



X_pred_pulse_interp=interp1(t_ds(1:length(X_pred_pulse_prepro)),X_pred_pulse_prepro,t_prepro,'cubic');

X_pred_pulse_smooth=smoothdata(X_pred_pulse_interp,'loess',500*1);

plot(t_prepro,X_pred_pulse_smooth,'LineWidth',2)

legend('true ECG','pulse ECG','reconstructed ECG');
hold off;
drawnow


%% Picchi

figure;

stem(time_ecg,ones(size(R_peak)),'LineWidth',2);
hold on;
stem(ecg_pulse_pred_prepro,0.5*ones(size(ecg_pulse_pred_prepro)),'LineWidth',2);
hold off;

drawnow
ylim([-0.05,1.05])

xlabel('seconds')
legend('true','estimated')

%% HRV 

figure

 plot(diff(time_ecg))
hold on
plot(diff(ecg_pulse_pred_prepro))

title("HRV SEeries")
legend('original','predicted')
hold off;
