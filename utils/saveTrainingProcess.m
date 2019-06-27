function saveTrainingProcess(filename, varargin)
% save the most recent training process into file
% filename: specify the filename to store
% varargin: specify extension if necessary. (default: png)

extension = '.png';
if nargin > 2
    error('Too many inputs.');
end
if nargin > 1
    extension = varargin{1};
end

% find training progress figure handle (due to the handle is hidden)
figs = findall(groot, 'type', 'figure');

% check folder existence
if ~exist('../figure', 'dir')
    mkdir('../figure');
end

% save training process into a png figure
saveas(figs(1), ['../figure/', filename, extension]);