function combinazioni = combinazioni_random(N,k,n_combinazioni)
    if(k>N)
        k=N;
    end
    for nn=1:n_combinazioni
        combinazioni(nn,:) = randperm(N,k);
    end
end