%**************************************************************************
% Simon Crete| 20047585 | 16sjcc
% PATH 828 PROJECT

% Computes days between date diagnosis and date of death or last followup 
% Creates NumDays collums and adds it to data before preprocessing
%**************************************************************************

% add directory with functions to path
addpath('../functions');

predictionData = anonymizedBrainMetsOutcomeData;
%importSurvivalData('anonymizedBrainMetsOutcomeData.xlsx');

% Variables to hold new computed values
NumDays = zeros(size(predictionData,1),1);
% Convert dates to days
for i = 1:size(predictionData,1)    
    NumDays(i,1) = daysact(predictionData{i,9},predictionData{i,8});
end

% Convert to table
NumDays = array2table(NumDays);

% Add number of days as a column
target = [predictionData NumDays];

% Write all_data table to file 
writetable(target, 'brainMets_features_survivalInDays.csv');

% plot GTV vs survival
%scatter(NumDays,rawSurvivalData{:,11}, 50);

% plot GTV vs GTV rad and categorize output
%scatter(rawSurvivalData{:,11},rawSurvivalData{:,12},69,NumDays)

