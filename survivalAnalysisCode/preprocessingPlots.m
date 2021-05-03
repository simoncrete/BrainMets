%**************************************************************************
% Simon Crete| 20047585 | 16sjcc
% PATH 828 PROJECT

% Script to create required statistics plots for preprocessing steps
%**************************************************************************

% add directory with functions to path
addpath('../functions');

% DATA MANIPULATION
%rawdataMat = cell2mat(celldata_raw(3:end,2:end));
new = num2cell(1:107);

% INITIAL VISUALIZATION
% NOTE: the alpha value is calculated inside plotScatterForData by a
% function call to computeAlphaOutliers with the length of the data passed
% into plotScatterForData.
plotScatterForData(mean(predictionData), 'Means Normalized Data', 'Count', new)

% IQR scatter plot for raw data
plotScatterForData(iqr(predictionData), 'IQR for Normalized Data', 'IQR', new);

% box plots for raw data
figure
boxplot(predictionData ,'Notch','on','Labels',new,'LabelOrientation','inline');
title('Boxplots of Normalized Features')
xlabel('Radiomic Features')
ylabel('Sum')

logtransform_processedData = log2(replaceZeros(predictionData, 'lowval'));

% visualize log2 transformation with boxplots
figure
boxplot(logtransform_processedData ,'Notch','on','Labels',new,'LabelOrientation','inline');
title('Boxplots of log transformed')
xlabel('Radiomic Features')
ylabel('Log2 normalized and transformed')






