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