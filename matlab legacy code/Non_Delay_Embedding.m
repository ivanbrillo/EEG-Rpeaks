function [X_nemb]=Non_Delay_Embedding(X,l)
    % X is the original multivariate series, row vector
    % l is the embedding dimension vector index
    X_nemb=X(l,:);
end