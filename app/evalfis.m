function varargout =  evalfis(varargin)
% EVALFIS    Perform fuzzy inference calculations.
%
%   OUTPUTS = EVALFIS(FIS,INPUTS) simulates the fuzzy inference system FIS
%   for the input data INPUTS and returns the output data OUTPUT. In this
%   case, EVALFIS uses a default option object returned by EVALFISOPTIONS
%   function. For a system with N input variables and L output variables,
%      * INPUTS is an M-by-N matrix, where each of the M rows is an input
%        vector.
%      * OUTPUTS is an M-by-L matrix, where each of the M rows is an output
%        vector for the corresponding input vector.
%   In this case, default evalfis options are used to evaluate FIS.
%   
%   OUTPUTS = EVALFIS(FIST,INPUTS) simulates FIST with input values INPUTS
%   and returns output values OUTPUTS. FIST is a scalar fistree object.
%   INPUTS is an M-by-N numeric matrix, where each of the M rows is an
%   input vector and N is the number of inputs of FIST. Input values in
%   each row  are specified according to the order of FIST inputs. Output
%   value OUTPUTS is M-by-L numeric matrix, where each of the M rows is an
%   output vector for the corresponding input vector and L is the number of
%   outputs of FIST. Output values in each row are returned according to
%   the order of outputs of FIST. In this case, default evalfis options are
%   used to evaluate each fuzzy inference system in FIST.
%            
%   OUTPUTS = EVALFIS(FIS,INPUTS,OPTIONS) simulates the fuzzy inference
%   system FIS using the option values specified in OPTIONS returned by
%   EVALFISOPTIONS function. For more information on creating options, see
%   EVALFISOPTIONS function.
%
%   OUTPUTS = EVALFIS(FIST,INPUTS,OPTIONS) simulates the fuzzy inference
%   system tree FIST using the option values specified in OPTIONS returned
%   by EVALFISOPTIONS function. For more information on creating options,
%   see EVALFISOPTIONS function.
%
%   [OUTPUTS,FUZZIFIEDINPUTS,RULEOUTPUTS,AGGREGATEDOUTPUTS,RULEFIRINGSTRENGTHS] = EVALFIS(...)
%   returns the following output arguments. If INPUTS is a matrix, these
%   output arguments correspond to the input vector in the last row of
%   INPUTS.
%      * FUZZIFIEDINPUTS: Result of evaluating the input values through the
%        membership functions. For a type-1 FIS, FUZZIFIEDINPUTS is
%        returned as a matrix of size Nr-by-N, where Nr is the number of
%        rules. For a type-2 FIS, FUZZIFIEDINPUTS is returned as a matrix
%        of size Nr-by-2*N. The first N columns and the last N columns
%        represent fuzzified input values using the upper membership
%        functions (UMFs) and lower membership functions (LMFs),
%        respectively.
%      * RULEOUTPUTS: Result of evaluating the rule consequents and then
%        applying implication operator on the consequent outputs. For a
%        Mamdani FIS, consequent outputs are fuzzy sets obtained by
%        evaluating the output values through the membership functions.
%        Hence, for a type-1 Mamdani FIS, RULEOUTPUTS is returned as a
%        matrix of size Ns-by-Nr*L, where Ns is the number of sample points
%        specified in OPTIONS. The first Nr columns correspond to the first
%        output, the next Nr columns correspond to the second output, and
%        so on. For a Sugeno FIS, consequent outputs are scalar values.
%        Hence, RULEOUTPUTS is returned as a matrix of size Nr-by-L. For a
%        type-2 FIS, RULEOUTPUTS is returned as a matrix of size
%        Ns-by-2*Nr*L. The first Nr*L columns correspond to the rule output
%        values generated using UMFs and the next Nr*L correspond to the
%        rule output values obtained using LMFs.
%      * AGGREGATEDOUTPUTS: Result of applying aggregation operator on the
%        rule outputs. For a type-1 Mamdani FIS, AGGREGATEDOUTPUTS is
%        returned as a matrix of size Ns-by-L matrix, where each column
%        contains the aggregated fuzzy set for an output. For a type-2 FIS,
%        AGGREGATEDOUTPUTS is returned as a matrix of size Ns-by-2*L, where
%        the first L columns correspond to the aggregated values obtained
%        using the UMFs and the last L columns correspond to the aggregated
%        values obtained using the LMFs.
%      * RULEFIRINGSTRENGTHS: Result of applying fuzzy connection operators
%        on rule antecedents. For a type-1 FIS, RULEFIRINGSTRENGTHS is
%        returned as a Nr-by-1 vector, where each row corresponds to the
%        combined membership value of a rule antecedent. For a type-2 FIS,
%        RULEFIRINGSTRENGTHS is returned as a Nr-by-2 vector, where the
%        first column represents the rule firing strength generated using
%        the UMFs and the second column corresponds to the rule firing
%        strength generated using the LMFs.
%   Multiple output arguments are not supported for a fistree object.
%
%   Example:
%       %% Evaluate fuzzy inference system
%       fis = readfis('tipper');
%       options = evalfisOptions;
%       options.NumSamplePoints = 50;
%       [y,irr,orr,arr,rfs] = EVALFIS(fis,[2 1; 4 9],options);
%
%       %% Evaluate tree of fuzzy inference systems
%       fis1 = mamfis('Name','fis1','NumInputs',2,'NumOutputs',1);
%       fis2 = mamfis('Name','fis2','NumInputs',2,'NumOutputs',1);
%       con = ["fis1/output1" "fis2/input1"];
%       fisT = fistree([fis1 fis2],con);
%       options = evalfisOptions('NumSamplePoints',50);
%       y = evalfis(fisT,[0.5 0.2 0.7],options);
%
%   See also FISTREE, EVALFISOPTIONS, READFIS, RULEVIEW, GENSURF.

%   Copyright 1994-2019 The MathWorks, Inc.

[varargout{1:nargout}] = fuzzy.internal.utility.evalfis(varargin{:});

end