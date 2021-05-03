%**************************************************************************
% Simon Crete| 20047585 | 16sjcc
% PATH 828 project

% Script to generate clustergrams
%**************************************************************************

% Data Import and Organization - Clinical Data
addpath('../functions');

% needed for formatting
counter = [1:19];
rows = [1:160];

% Remove Zeroes and Log transform data
zeroremoved = replaceZeros(predictionData1,'lowval');
mat_log = log2(zeroremoved);

% Median Center the Data
mat_medcent = (mat_log - median(mat_log));

% Check that median has been set to 0 for all samples 
median(mat_medcent); 

% Hierarchical Clustering
% Set sample labels
sample_label_TS = counter;

% Row Labels 
row_label = yearBinary;

% Clustergram 1 using default parameters 
clustergram_TS = clustergram(mat_medcent, 'RowLabels',row_label,...
                               'ColumnLabels', sample_label_TS,... 
                               'RowPDist', 'euclidean',...
                               'ColumnPDist', 'spearman', 'Colormap', 'jet',...
                               'DisplayRange', 7, 'LabelsWithMarkers', true);
                           
% Clustergram 2 - change similarity measures to both rows and columns being euclidean 
clustergram_TS_2 = clustergram(mat_medcent, 'RowLabels',row_label,...
                               'ColumnLabels', sample_label_TS,... 
                               'RowPDist', 'euclidean',...
                               'ColumnPDist', 'euclidean', 'Colormap', 'jet',...
                               'DisplayRange', 7, 'LabelsWithMarkers', true);

% Clustergram 3 - change linkage type to weighted from default of 'average'
clustergram_TS_3 = clustergram(mat_medcent, 'RowLabels',row_label,...
                               'ColumnLabels', sample_label_TS,... 
                               'RowPDist', 'euclidean',...
                               'ColumnPDist', 'euclidean', 'Colormap', 'jet',...
                               'Linkage', 'weighted',...
                               'DisplayRange', 7, 'LabelsWithMarkers', true);                                                  
                                                     