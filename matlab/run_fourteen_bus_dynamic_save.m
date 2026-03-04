%% Run fourteen_bus_dynamic (5 s) and save all results to CSV, TXT, PNG
% Usage: run('run_fourteen_bus_dynamic_save.m')
% Outputs: results/fourteen_bus_dynamic/*.csv, *.txt, *.png

scriptDir = fileparts(mfilename('fullpath'));
if ~isempty(scriptDir), cd(scriptDir); end

outDir = fullfile(scriptDir, 'results', 'fourteen_bus_dynamic');
if ~isfolder(outDir), mkdir(outDir); end

model = 'fourteen_bus_dynamic';
fprintf('Running fourteen_bus_dynamic (5 s)...\n');
load_system(model);
set_param(model, 'UnconnectedOutputMsg', 'none');
% Log full 0-5 s (align with Python); disable data point limit
try
  set_param(model, 'LimitDataPoints', 'off');
catch
end
scopes = find_system(model, 'BlockType', 'Scope');
for k = 1:numel(scopes)
  try
    set_param(scopes{k}, 'LimitDataPoints', 'off');
  catch
  end
end
sim(model);

% --- Save ScopeBus1..14 to CSV (one file per bus) and one combined summary ---
summaryPath = fullfile(outDir, 'summary.txt');
fid = fopen(summaryPath, 'w');
fprintf(fid, 'fourteen_bus_dynamic results\n');
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

% --- Derive ROCOF_max and f_min from voltage (same definition as Python) ---
f_nominal = 50;
fs_obs = 12;
obsRows = cell(14, 1);
obsCount = 0;
for i = 1:14
    name = sprintf('ScopeBus%d', i);
    if ~exist(name, 'var'), continue; end
    d = evalin('base', name);
    t = d.time(:);
    if ~isfield(d, 'signals') || isempty(d.signals), continue; end
    vals = d.signals(1).values;
    if size(vals, 2) < 3, continue; end
    va = vals(:,1); vb = vals(:,2); vc = vals(:,3);
    v_alpha = (2/3)*(va - 0.5*vb - 0.5*vc);
    v_beta = (1/sqrt(3))*(vb - vc);
    phi = unwrap(atan2(v_beta, v_alpha));
    if numel(t) < 2
        rocof_max = NaN; f_min = NaN;
    else
        f_inst = gradient(phi, t) / (2*pi);
        delta_f = f_inst - f_nominal;
        t_min = min(t); t_max = max(t);
        n_obs = max(2, round((t_max - t_min) * fs_obs));
        t_obs = linspace(t_min, t_max, n_obs);
        delta_f_obs = interp1(t, delta_f, t_obs);
        h_obs = (t_max - t_min) / (n_obs - 1);
        rocof = gradient(delta_f_obs, h_obs);
        rocof_max = max(abs(rocof));
        f_min = f_nominal + min(delta_f_obs);
    end
    obsCount = obsCount + 1;
    obsRows{obsCount} = [i, rocof_max, f_min];
end
if obsCount > 0
    obsCsv = fullfile(outDir, 'observation_from_voltage.csv');
    fidO = fopen(obsCsv, 'w');
    fprintf(fidO, 'bus,ROCOF_max,f_min\n');
    for k = 1:obsCount
        r = obsRows{k};
        fprintf(fidO, '%d,%.10g,%.10g\n', r(1), r(2), r(3));
    end
    fclose(fidO);
    fid = fopen(summaryPath, 'a');
    fprintf(fid, 'Derived (same as Python): f_nominal=%g Hz, fs=%g Hz\n', f_nominal, fs_obs);
    fprintf(fid, 'bus\tROCOF_max (Hz/s)\tf_min (Hz)\n');
    for k = 1:obsCount
        r = obsRows{k};
        fprintf(fid, '%d\t%.6f\t%.6f\n', r(1), r(2), r(3));
    end
    fclose(fid);
end

% --- Plots (PNG): one figure per bus + overview (larger, less condensed) ---
figPos = [0, 0, 900, 500];  % wide figure for dynamics
for i = 1:14
    name = sprintf('ScopeBus%d', i);
    if ~exist(name, 'var'), continue; end
    d = evalin('base', name);
    t = d.time(:);
    f = figure('Visible', 'off', 'Position', figPos, 'PaperPositionMode', 'auto');
    if isfield(d, 'signals') && ~isempty(d.signals)
        vals = d.signals(1).values;
        if size(vals, 2) > 1
            plot(t, vals, 'LineWidth', 1);
        else
            plot(t, vals(:), 'LineWidth', 1);
        end
    end
    xlabel('Time (s)', 'FontSize', 12); ylabel('Signal', 'FontSize', 12);
    title(sprintf('Fourteen\\_bus\\_dynamic — Bus %d', i), 'FontSize', 14);
    set(gca, 'FontSize', 11); grid on;
    saveas(f, fullfile(outDir, [name, '.png']));
    close(f);
end

% Overview: 7x2 subplots (taller subplots), larger figure
f = figure('Visible', 'off', 'Position', [0, 0, 1200, 1400], 'PaperPositionMode', 'auto');
for i = 1:14
    name = sprintf('ScopeBus%d', i);
    if ~exist(name, 'var'), continue; end
    d = evalin('base', name);
    subplot(7, 2, i);
    t = d.time(:);
    if isfield(d, 'signals') && ~isempty(d.signals)
        vals = d.signals(1).values;
        if size(vals, 2) >= 1
            plot(t, vals(:,1), 'LineWidth', 1);
        end
    end
    title(sprintf('Bus %d', i), 'FontSize', 12);
    xlabel('t (s)', 'FontSize', 10); set(gca, 'FontSize', 10); grid on;
end
sgtitle('fourteen\_bus\_dynamic — all buses', 'FontSize', 14);
set(f, 'PaperPosition', [0 0 12 14]);
saveas(f, fullfile(outDir, 'all_buses.png'));
close(f);

close_system(model, 0);
fprintf('Results saved to: %s\n', outDir);
