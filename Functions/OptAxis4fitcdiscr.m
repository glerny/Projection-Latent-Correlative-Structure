%% DESCRIPTION
% OptAxis4fitcdiscr is used to select 2 to 5 axes that provide the best
% combination for clustering or classification. It has been designed for
% PLCS but should work for other data reduction approaches. In input,
% Structure is the data structure, a [MxN] matrix,  with, in lines, the
% axes and, in columns, the samples. Y in a [N] vector of tag that defined
% the belonging of each sample to a given group. nbrAxis (2 to 5) id the
% number of final axes that aim to be obtained and DiscrimType is either
% 'linear' or 'quadratic'. In the output, an 8XM table is obtained and
% contained classification results using a soft discriminant analysis
% classification(d^2 <= 3). The first column (AxisXomb) indicate the
% combination of Axes under investigation. nm is the number of samples
% classified into two or more categories; ns is the number of samples
% accurately classified; no is the number of unclassified samples; ms is
% the number of samples classified to the wrong category; tb is nm+ms. Best
% selection axis is the one with the lowest tb. DAModel is the
% classification discriminant model.
%
%% REFERENCES
% If you are using OptAxis4fitcdiscr, please use the following reference(s):
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

function Results  = OptAxis4fitcdiscr(XTrain, YTrain, XTest, YTest, nbrAxis, DiscrimType)

% TODO: Check data and convert categorial Y.
% TOCHECK: What if Y not ordered
YCat   = unique(YTrain);
nGps   = numel(YCat);
Alfa4Outlier = 0.01;
Alfa4SoftDA = 0.05;

% 1. Find all possible combination of Axis
AllCombi = [];
if nbrAxis == 2
    for ii = 1:size(XTrain, 2)
        for jj = ii+1:size(XTrain, 2)
            AllCombi(end+1, :) = [ii, jj];
        end
    end
    
elseif nbrAxis == 3
    for ii = 1:size(XTrain, 2)
        for jj = ii+1:size(XTrain, 2)
            for kk = jj+1:size(XTrain, 2)
                AllCombi(end+1, :) = [ii, jj, kk];
            end
        end
    end
    
elseif nbrAxis == 4
    for ii = 1:size(XTrain, 2)
        for jj = ii+1:size(XTrain, 2)
            for kk = jj+1:size(XTrain, 2)
                for ll = kk+1:size(XTrain, 2)
                    AllCombi(end+1, :) = [ii, jj, kk, ll];
                end
            end
        end
    end
    
elseif nbrAxis == 5
    for ii = 1:size(XTrain, 2)
        for jj = ii+1:size(XTrain, 2)
            for kk = jj+1:size(XTrain, 2)
                for ll = kk+1:size(XTrain, 2)
                    for mm = ll+1:size(XTrain, 2)
                        AllCombi(end+1, :) = [ii, jj, kk, ll, mm];
                    end
                end
            end
        end
    end
end

% TODO: trap and verify warning

for nc = 1:size(AllCombi, 1)
    X = XTrain(:, AllCombi(nc,:));
    
    try
        %TODO: CHeck and verify errors
        Mdl = fitcdiscr(X, YTrain, 'DiscrimType',DiscrimType);
        mhd = mahal(Mdl, X);
        ido = false(size(YTrain));
        
        SoftClustDA = [];
        for ii = 1:nGps
            tgt = find(YTrain==ii);
            ido(tgt(isoutlier(mhd(tgt,ii)))) = true;
            SoftClustDA(ii) = sqrt(chi2inv(1-Alfa4SoftDA, ...
                sum(~isoutlier(mhd(tgt,ii))) - 1));
        end
        
        X_c = X(~ido, :);
        Y_c = YTrain(~ido);
        Mdl = fitcdiscr(X_c, Y_c, 'DiscrimType',DiscrimType);
        mhd = mahal(Mdl, X);
        
        
        
        prov = Mdl.mahal(Mdl.Mu);
        ClustersResolution = min(prov, prov');
        
        % Hard Correlation
        ypred = predict(Mdl, XTest(:, AllCombi(nc,:)));
        C_hard = confusionmat(YTest, ypred);
        
        % Soft Correlation
        mahalDist = mahal(Mdl, XTest(:, AllCombi(nc,:)));
        mahalDist = mahalDist./SoftClustDA;
        [~, YHat] = min(mahalDist, [], 2); %smallest Mah Dist
        YHat(sum(mahalDist < 1, 2) == 0) = nGps+1; % not classified with d^2 <= 3;
        YHat(sum(mahalDist < 1, 2) > 1) = nGps+2; % multiple classification;
        C_soft = confusionmat(YTest, YHat);
        
        % Recording
        AxisComb{nc, 1} = AllCombi(nc,:);
        HardConfMatrix{nc, 1} = C_hard;
        n_hard(nc, 1) = sum(diag(C_hard));
        SoftConfMatrix{nc, 1} = C_soft;
        n_soft(nc, 1) = sum(diag(C_soft));
        TagsInC_soft{nc, 1} = unique(YHat);
        ThresSoftDA{nc, 1} = SoftClustDA;
        DAModel{nc, 1} = Mdl;
        Resolution{nc, 1} = ClustersResolution;
        MinResolution(nc, 1) = min(ClustersResolution(ClustersResolution ~=0));
        
    catch EM
        
        % TODO: Check error
        AxisComb{nc, 1} = AllCombi(nc,:);
        HardConfMatrix{nc, 1} = [];
        n_hard(nc, 1) = NaN;
        SoftConfMatrix{nc, 1} = [];
        n_soft(nc, 1) = NaN;
        TagsInC_soft{nc, 1} = [];
        ThresSoftDA{nc, 1} = [];
        DAModel{nc, 1} = {};
        Resolution{nc, 1} = [];
        MinResolution(nc, 1) = NaN;
        continue
    end
    
end
Results = table(AxisComb, HardConfMatrix, n_hard, SoftConfMatrix, n_soft, TagsInC_soft, ThresSoftDA, DAModel, Resolution, MinResolution);
Results.ID = (1:size(Results,1))';
