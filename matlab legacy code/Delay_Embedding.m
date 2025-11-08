function [X_emb]=Delay_Embedding(X,m)
    % X is the original monovariate series
    % m is the embedding dimension
    X=reshape(X,1,[]); 
    N=length(X);
    for ii=1:m
    X_emb(ii,:)=X(ii:N-m+ii);
    end
end