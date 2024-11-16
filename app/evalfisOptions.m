function options = evalfisOptions(varargin)
%% EVALFISOPTIONS Creates options used by EVALFIS function.
%
%    EVALFISOPTIONS Creates options with default values.
%
%    EVALFISOPTIONS(name1,value1, ...) Creates options with the
%    specified parameter name/value pairs.
%
%    You can specify the following parameters:
%
%        'NumSamplePoints' - Number of sample points for which to evaluate
%        the membership functions over the output range, specified as an
%        integer grater than 1.The default value of NumSamplePoints is 101.
%
%        'OutOfRangeInputValueMessage' - Diagnostic type for out of input
%        range values, specified as "none", "warning", or "error". The
%        default value is "warning".
%
%        'NoRuleFiredMessage' - Diagnostic type for no-rule-fired
%        condition, specified as "none", "warning", or "error". The default
%        value is "warning".
%
%        'EmptyOutputFuzzySetMessage' - Diagnostic type for empty output
%        fuzzy sets, specified as "none", "warning", or "error". The
%        default value is "warning".
%
%    Examples:
%      %% Create options with default values
%      options = evalfisOptions;
%      evalfis(readfis('tipper'),[0 0],options);
%
%      %% Specify name/value pairs
%      options = evalfisOptions('NumSamplePoints',50);
%      evalfis(readfis('tipper'),[0 0],options);
%
%      %% Update existing option values
%      options = evalfisOptions;
%      option.NumSamplePoints = 50;
%      evalfis(readfis('tipper'),[0 0],options);
%
%    See also evalfis

%  Copyright 2017-2018 The MathWorks, Inc.

options = fuzzy.evalfis.EvalFISOptions(varargin{:});

end