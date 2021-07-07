%% DESCRIPTION
% Projection by Latent Correlative Structure (PLCS) is a dimension
% reduction strategy developed from spectroscopic-like data. In input, the
% function needs two matrixes: Data and Ref. Data is [mxn] matrix of n
% spectra, aligned to a common axis, and organised in columns (each column
% is a different spectrum), Ref is the [mxo] matrix of o references
% spectra. With  PLCS, each latent axis is the relative similarity of every
% spectrum to a pair of references. The latent structure is the relative
% similarity of every spectrum to every possible pairing of Ref spectra. In
% the output, alfa is the [oxn] latent structure. Samples are organised in
% lines and each column corresponds to the relative similarity of each
% spectrum to a pairing of reference spectra. Mapping is a [ox2] matrix
% that indicates the reference spectra used for each latent axis (e.g. [1
% 5] indicates that this axis is the relative similarity for every spectrum
% using the 1st and 5th reference spectra)
%
%% REFERENCES
% If you are using PLCS_proj, please use the following reference(s):
% * 
%
%% ACKNOWLEDGMENT 
% This work was financially supported by the projects: (i)
% UID/EQU/00511/2019 - Laboratory for Process Engineering, Environment,
% Biotechnology and Energy – LEPABE funded by national funds through
% FCT/MCTES (PIDDAC); (ii) POCI-01-0145-FEDER-029702 funded by FEDER funds
% through COMPETE2020 – Programa Operacional Competitividade e
% Internacionalização (POCI) and by national funds (PIDDAC) through
% FCT/MCTES. This algorithms is based upon work from COST Action CA 16215,
% supported by COST (European Cooperation in Science and Technology)
% www.cost.eu
%
%% COPYRIGHT
% Copyright BSD 3-Clause License Copyright 2021 G. Erny
% (guillaume@fe.up.pt), FEUP, Porto, Portugal
%
%%

function [alfa, RefCombination] = PLCS_proj(Data, Ref)

narginchk(2, 2)
% TODO: Check entries
% TODO: Verify variables

nbrRef = size(Ref, 2);
RefCombination = [];
for ii = 1:nbrRef
    for jj = ii+1:nbrRef
        RefCombination(end+1, :) = [ii jj];
    end
end

% TODO: Improve vectorization
for ii = 1:size(RefCombination, 1)
    P = pdist2(Data', Ref(:, RefCombination(ii,:))', 'correlation');
    Xc = 1 - corrcoef(Ref(:, RefCombination(ii,:)));
    V1 = Xc(1,[1, 2]);
    V2 = Xc(1,[2, 1]);
    for jj = 1:size(P,1)
        v = (V2-V1)/norm(V2-V1); %// normalized vector from V1 to V2
        Q(jj,:) = dot(P(jj, :)-V1,v)*v+V1; %// projection of P onto line from V1 to V2
        dist(jj) = norm(P(jj,:)-Q(jj,:));
        alfa(jj,ii) = (Q(jj, 1)-V1(1))/(V2(1)-V1(1));
    end
end

