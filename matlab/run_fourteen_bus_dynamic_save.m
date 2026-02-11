%% Run Fourteen_bus_dynamic (5 s) and save all results to CSV, TXT, PNG
% Usage: run('run_fourteen_bus_dynamic_save.m')
% Outputs: results/fourteen_bus_dynamic/*.csv, *.txt, *.png

scriptDir = fileparts(mfilename('fullpath'));
if ~isempty(scriptDir), cd(scriptDir); end

outDir = fullfile(scriptDir, 'results', 'fourteen_bus_dynamic');
if ~isfolder(outDir), mkdir(outDir); end

model = 'Fourteen_bus_dynamic';
fprintf('Running Fourteen_bus_dynamic (5 s)...\n');
load_system(model);
set_param(model, 'UnconnectedOutputMsg', 'none');
sim(model);

% --- Save ScopeBus1..14 to CSV (one file per bus) and one combined summary ---
summaryPath = fullfile(outDir, 'summary.txt');
fid = fopen(summaryPath, 'w');
fprintf(fid, 'Fourteen_bus_dynamic results\n');
fprintf(fid, 'Model: %s\n', model);
fprintf(fid, 'Saved: %s\n\n', datestr(now));

for i = 1:14
    name = sprintf('ScopeBus%d', i);
    if ~exist(name, 'var'), continue; end
    d = evalin('base', name);
    t = d.time(:);
    nt = numel(t);
    fprintf(fid, '%s: time [%g, %g] s, %d points\n', name, min(t), max(t), nt);

    % CSV: time + all signal columns
    if isfield(d, 'signals') && ~isempty(d.signals)
        vals = d.signals(1).values;
        if size(vals, 1) == nt
            M = [t, vals];
        else
            M = [t, vals(:)];
        end
        writematrix(M, fullfile(outDir, [name, '.csv']));
    end
end
fclose(fid);

% --- Plots (PNG): one figure per bus + one overview ---
for i = 1:14
    name = sprintf('ScopeBus%d', i);
    if ~exist(name, 'var'), continue; end
    d = evalin('base', name);
    t = d.time(:);
    f = figure('Visible', 'off');
    if isfield(d, 'signals') && ~isempty(d.signals)
        vals = d.signals(1).values;
        if size(vals, 2) > 1
            plot(t, vals);
        else
            plot(t, vals(:));
        end
    end
    xlabel('Time (s)'); ylabel('Signal');
    title(sprintf('Fourteen\\_bus\\_dynamic — Bus %d', i));
    saveas(f, fullfile(outDir, [name, '.png']));
    close(f);
end

% Overview: all buses in one figure (subplots)
f = figure('Visible', 'off');
for i = 1:14
    name = sprintf('ScopeBus%d', i);
    if ~exist(name, 'var'), continue; end
    d = evalin('base', name);
    subplot(4, 4, i);
    t = d.time(:);
    if isfield(d, 'signals') && ~isempty(d.signals)
        vals = d.signals(1).values;
        if size(vals, 2) >= 1
            plot(t, vals(:,1));
        end
    end
    title(sprintf('Bus %d', i));
    xlabel('t (s)');
end
sgtitle('Fourteen\_bus\_dynamic — all buses');
saveas(f, fullfile(outDir, 'all_buses.png'));
close(f);

fprintf('Results saved to: %s\n', outDir);
