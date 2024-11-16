function fisout = getFISCodeGenerationData(fisin,varargin)
% GETFISCODEGENERATIONDATA Creates a homogeneous FIS structure
%
%   FISOUT = GETFISCODEGENERATIONDATA(FISIN) Creates a homogeneous fuzzy
%   inference system structure FISOUT from FISIN, where FISIN is 
%     - a fuzzy inference system object.
%   If FISIN is a fuzzy inference system object, the structure arrays in 
%   FISOUT contain same size field values.
%
%     - a fuzzy inference system tree object.
%   If FISIN is a fuzzy inference system tree object, the structure array
%   in FISOUT contain 4 fields: FIS, Connections, Outputs and
%   EvaluationFcn. Each element in the FIS field is a structure containing
%   same size field values corresponding to each fuzzy inference system in
%   the tree. In case of a FIS tree input, parameter FuzzySetType will be
%   ignored.
%
%     - a FIS file name to be loaded from the disk.
%   If FISIN is a file name, it is assumed to contain membership functions
%   of type 1 fuzzy sets. The structure arrays in FISOUT contain same size
%   field values.
%
%   FISOUT = GETFISCODEGENERATIONDATA(FISIN,'FuzzySetType',TYPE) Creates a
%   homogeneous fuzzy inference system structure FISOUT from FISIN using
%   the specified type of membership function. If TYPE is:
%     - "type1" use type-1 membership functions. (default)
%     - "type2" use type-2 membership functions.
%   TYPE must match the type of membership functions in FISIN.
%
%   Examples
%
%   %% Get homogeneous FIS from an object
%   fis = GETFISCODEGENERATIONDATA(mamfis('NumInputs',1,'NumOutputs',1));
%
%   %% Load homogeneous FIS from a FIS file name
%   fis = GETFISCODEGENERATIONDATA('tipper.fis');
%
%   %% Get output structure from a FIS tree object
%   fis = GETFISCODEGENERATIONDATA(FISTREE);
%
%   See also
%     mamfis, sugfis, mamfistype2, sugfistype2, fistree

%   Copyright 2018-2022 The MathWorks, Inc.

narginchk(1,Inf)

p = inputParser;
p.addRequired('FISIN',@validateFISInput)
p.addParameter("FuzzySetType",'',@validateFuzzySetType)
p.parse(fisin,varargin{:})

if isa(fisin,'fistree')
    fisout = createCodeGenerationData(fisin);
    return
end

if isa(p.Results.FISIN,'FuzzyInferenceSystem')
    if ~isempty(p.Results.FuzzySetType)
        if (p.Results.FuzzySetType=="type1" && (isa(fisin,'mamfistype2')||isa(fisin,'sugfistype2'))) || ...
                (p.Results.FuzzySetType=="type2" && (isa(fisin,'mamfis')||isa(fisin,'sugfis')))
            error(message("fuzzy:general:errGetFISCodeGenerationData_fisIsNotCorrectType"))
        end        
    end
    fisout = fuzzy.internal.utility.convertToHomogenousFISStruct(fisin);
else
    if isempty(p.Results.FuzzySetType) || p.Results.FuzzySetType=="type1"
        fisout = fuzzy.internal.codegen.readFISAsHomogenousStructure(fisin);
    else
        fisout = fuzzy.internal.codegen.readType2FISAsHomogenousStructure(fisin);
    end
end

fuzzy.internal.utility.writeCustomFunctions(fisout);
end
%% Helper functions -------------------------------------------------------
function validateFISInput(value)
validateattributes(value,{'char','string','FuzzyInferenceSystem','fistree'},{'nonempty'},'','FIS input')
if ischar(value) || isstring(value)
    fuzzy.internal.utility.validCharOrString('FIS file name',value);
else
    validateattributes(value,{'FuzzyInferenceSystem','fistree'},{'scalar'},'','FIS object')
end
end

function validateFuzzySetType(value)
fuzzy.internal.utility.validCharOrString('Fuzzy set type',value);
if ~any(value==["type1" "type2"])
    error(message("fuzzy:general:errGetFISCodeGenerationData_invalidFSType"))
end
end