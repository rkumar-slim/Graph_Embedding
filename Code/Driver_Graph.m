clear all;clc;
addpath(genpath('Code/mbin/'));
n               = [101 301];
h               = [50 50];
z               = [0:n(1)-1]*h(1);
x               = [0:n(2)-1]*h(2);
[zz,xx]         = ndgrid(z,x);
vfun            = @(v0,alpha)(v0 + alpha*zz)+ 400*exp(-1e-6*(zz - 2000).^2);
%% Generate data
model.o         = [0 0];
model.d         = h;
model.n         = n;
model.nb        = [60 60;60 60];
model.xsrc      = x(1:2:end); %x(1:2:end);
nsrc            = length(model.xsrc);
model.zsrc      = [h(1)];
model.xrec      = x(1:2:end); %x(1:2:end);
nrec            = length(model.xrec);
model.zrec      = [h(1)];
model.f0        = 10;   %center frequency. Curt origianl set to 15
model.t0        = 0;
Q               = eye(nsrc);
model.unit      = 's2/km2';
model.nrec      = nrec;
model.nsrc      = nsrc;
model.freq      = 5;
v               = vfun(2000,0.5);
m               = 1e6./v(:).^2;
model.W         = 1;
Dtrue           = F(m, Q, model);
vs              = [1000:100:3000];
for i = 1:length(vs);
    v0               = vfun(vs(i),0.5);
    m0               = 1e6./v0(:).^2;
    Dapp             = F(m0, Q, model);
    Fs(i)            = norm(Dtrue(:)-Dapp(:));
end
Dtrue         = gather(reshape(Dtrue,nrec,nsrc));
Dtrue         = Dtrue./norm(Dtrue(:));
%% Graph Embedding
sigma         = 0.3;
gassfunc      = @(x1,x2)exp(-norm(x1-x2)^2/(2*sigma^2));
% construct weighted graph G or  (no self loop so diagonal is zero)
K             = zeros(nrec,nsrc);
for i = 1:nsrc
    for j = 1:nsrc
        if i==j
%             K(i,j) = 0;
        else
            K(i,j)      = gassfunc(Dtrue(:,i),Dtrue(:,j));
        end
    end
end

% Markov chain construction, number of non zeros in each rows of K that will make the D
D1      = sum(K,2);
% do normlaization
steps   = 4;
numsvd  = 10;
P       = diag(1./D1)*K;
% perform SVD
[U,S,V] = svds(P,numsvd);
S       = S.^steps;
%%
for l = 1:length(vs);
    v0               = vfun(vs(l),0.5);
    m0               = 1e6./v0(:).^2;
    Dapp             = F(m0, Q, model);
    Dapp             = gather(reshape(Dapp,nrec,nsrc));
    Dapp             = Dapp./norm(Dapp(:));
    % construct weighted graph G or  (no self loop so diagonal is zero)
    K                = zeros(nrec,nsrc);
    for i = 1:nsrc
        for j = 1:nsrc
            if i==j
                K(i,j) = 0;
            else
                K(i,j)      = gassfunc(Dapp(:,i),Dapp(:,j));
            end
        end
    end
    % Markov chain construction, number of non zeros in each rows of K that will make the D
    D1      = sum(K,2);
    % do normlaization
    P       = diag(1./D1)*K;
    % perform SVD
    [U1,S1,V1] = svds(P,numsvd);
    S1         = S1.^steps;
    Fobj(l)    = norm(U*S*V'-U1*S1*V1','fro');
end
