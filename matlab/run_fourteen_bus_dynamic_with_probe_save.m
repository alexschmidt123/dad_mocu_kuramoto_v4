%% Run fourteen_bus_dynamic_probe (5 s, with Hann-window probe) and save results
% Usage: run('run_fourteen_bus_dynamic_with_probe_save.m')
% - Adds probe blocks to the model if not already present (Hann window A=0.2, Tp=2 s)
% - Runs simulation, saves to results/fourteen_bus_dynamic_probe/
%
% IMPORTANT: The probe signal is generated but must be INJECTED into the grid to see
% probe effect (disturbance 0-2 s, recovery 2-5 s). Without injection, results match
% the no-probe run. See matlab/instruction.md § "Probe injection (active probing at bus 1)".

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
addpath(scriptDir);
if ~isempty(scriptDir), cd(scriptDir); end

outDir = fullfile(scriptDir, 'results', 'fourteen_bus_dynamic_probe');
if ~isfolder(outDir), mkdir(outDir); end

model = 'fourteen_bus_dynamic_probe';
% Add probe at model root (avoids subsystem path differences across MATLAB versions)
root = model;
mdlPath = fullfile(scriptDir, [model '.mdl']);

% Load model (ensure fourteen_bus_dynamic_probe.mdl exists; exist returns 4 for .mdl)
if exist(mdlPath, 'file') == 0
    error('fourteen_bus_dynamic_probe.mdl not found at %s', mdlPath);
end
load_system(model);

% Add Hann-window probe blocks if not already present
if ~block_exists([root '/HannProbe'])
    add_block('simulink/Sources/Constant', [root '/ProbeA']);
    add_block('simulink/Sources/Constant', [root '/ProbeTp']);
    add_block('simulink/Sources/Clock', [root '/ProbeClock']);
    add_block('simulink/Signal Routing/Mux', [root '/ProbeMux']);
    add_block('simulink/User-Defined Functions/Fcn', [root '/HannProbe']);
    add_block('simulink/Sinks/Out1', [root '/ProbeOut']);
    set_param([root '/ProbeA'], 'Value', '0.2');
    set_param([root '/ProbeTp'], 'Value', '2');
    set_param([root '/ProbeMux'], 'Inputs', '3');
    set_param([root '/HannProbe'], 'Expr', '0.5*u(1)*(1-cos(2*pi*u(2)/u(3)))*(u(2)<=u(3))');
    set_param([root '/ProbeOut'], 'Port', '1');
    set_param([root '/ProbeA'],    'Position', [80,  40, 120, 60]);
    set_param([root '/ProbeTp'],   'Position', [80,  80, 120, 100]);
    set_param([root '/ProbeClock'],'Position', [80, 120, 120, 140]);
    set_param([root '/ProbeMux'],  'Position', [180, 60, 200, 120]);
    set_param([root '/HannProbe'], 'Position', [260, 70, 380, 110]);
    set_param([root '/ProbeOut'],  'Position', [430, 85, 450, 95]);
    add_line(root, 'ProbeA/1', 'ProbeMux/1');
    add_line(root, 'ProbeClock/1', 'ProbeMux/2');
    add_line(root, 'ProbeTp/1', 'ProbeMux/3');
    add_line(root, 'ProbeMux/1', 'HannProbe/1');
    add_line(root, 'HannProbe/1', 'ProbeOut/1');
    set_param(model, 'SaveOutput', 'on');
    save_system(model);
    fprintf('Probe blocks added to %s.\n', model);
end

% Try to add probe injection at bus 1 (Controlled Current Source driven by HannProbe)
% Library path depends on MATLAB version; try common variants.
if ~block_exists([root '/ProbeInjection'])
  libPaths = {
    'powerlib/Electrical/Sources/Controlled Current Source',
    'powerlib/Sources/Controlled Current Source',
    [ 'powerlib/Electrical', char(10), 'Sources/Controlled Current Source' ],
  };
  added = false;
  for k = 1:numel(libPaths)
    try
      add_block(libPaths{k}, [root '/ProbeInjection']);
      added = true;
      break;
    catch
    end
  end
  if added
    set_param([root '/ProbeInjection'], 'Position', [260, 200, 320, 240]);
    try
      set_param([root '/ProbeInjection'], 'Initialize', 'off');
    catch
    end
    try
      set_param([root '/ProbeInjection'], 'Initial amplitude', '0');
    catch
    end
    add_line(root, 'HannProbe/1', 'ProbeInjection/1');
    save_system(model);
    fprintf('ProbeInjection block added and driven by HannProbe.\n');
  end
end

% Wire ProbeInjection into grid: + to Bus 1 (one phase), - to Ground
% SPS requires a snubber (high-R branch in parallel) when a current source is in series with inductors.
% Bus 1 in the model is named "Bus 1 " (trailing space). Add Ground, snubber, and connect.
probeWired = false;
if block_exists([root '/ProbeInjection'])
  bus1Blk = [root '/Bus 1 '];  % trailing space per .mdl
  if block_exists(bus1Blk)
    % Add Ground for current source return path
    gndBlk = [root '/ProbeGround'];
    if ~block_exists(gndBlk)
      gndLibs = { 'powerlib/Elements/Ground', 'powerlib/Utilities/Ground', ...
                  'sps_lib/Elements/Ground', 'sps_lib/Utilities/Ground' };
      for k = 1:numel(gndLibs)
        try
          add_block(gndLibs{k}, gndBlk);
          set_param(gndBlk, 'Position', [260, 260, 300, 290]);
          break;
        catch
        end
      end
    end
    % Add high-value resistive snubber in parallel with current source (required by SPS solver)
    snubBlk = [root '/ProbeSnubber'];
    if block_exists(gndBlk) && ~block_exists(snubBlk)
      snubLibs = { 'powerlib/Elements/Series RLC Branch', 'sps_lib/Elements/Series RLC Branch' };
      for k = 1:numel(snubLibs)
        try
          add_block(snubLibs{k}, snubBlk);
          set_param(snubBlk, 'Position', [320, 200, 380, 240]);
          set_param(snubBlk, 'BranchType', 'R');
          set_param(snubBlk, 'Resistance', '1e6');
          break;
        catch
        end
      end
    end
    if block_exists(gndBlk)
      phB = get_param(bus1Blk, 'PortHandles');
      phG = get_param(gndBlk, 'PortHandles');
      gndPort = [];
      if ~isempty(phG.LConn), gndPort = phG.LConn(1); elseif ~isempty(phG.RConn), gndPort = phG.RConn(1); end
      % Add snubber first (high-R in parallel) so current source is not in series with inductor
      if block_exists(snubBlk) && ~isempty(phB.LConn) && ~isempty(gndPort)
        try
          phS = get_param(snubBlk, 'PortHandles');
          if ~isempty(phS.LConn) && ~isempty(phS.RConn)
            add_line(root, phB.LConn(1), phS.LConn(1));
            add_line(root, phS.RConn(1), gndPort);
          end
        catch ME
          if ~contains(ME.message, 'already has a line')
            fprintf(2, 'Snubber connection: %s\n', ME.message);
          end
        end
      end
      % Connect ProbeInjection (+ to Bus 1, - to Ground) if not already connected
      try
        phP = get_param([root '/ProbeInjection'], 'PortHandles');
        if ~isempty(phP.LConn) && ~isempty(phP.RConn) && ~isempty(phB.LConn) && ~isempty(gndPort)
          add_line(root, phP.LConn(1), phB.LConn(1));
          add_line(root, phP.RConn(1), gndPort);
          probeWired = true;
          save_system(model);
          fprintf('ProbeInjection wired: + to Bus 1, - to Ground (with snubber). Probe is active.\n');
        end
      catch ME
        if contains(ME.message, 'already connected') || contains(ME.message, 'already has a line')
          probeWired = true;
          save_system(model);
          fprintf('ProbeInjection already wired (snubber added if missing). Probe is active.\n');
        else
          fprintf(2, 'Could not wire ProbeInjection: %s\n', ME.message);
        end
      end
    end
    if ~probeWired
      fprintf('  Connect ProbeInjection + and - to Bus 1 and ground in Simulink (see instruction.md), then save and re-run.\n');
    end
  end
end

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
% Warn if probe is not injected (no disturbance in grid)
if ~probeWired && block_exists([root '/ProbeInjection'])
  fprintf(2, 'WARNING: ProbeInjection is not wired to the grid (only signal connected).\n');
  fprintf(2, '         Connect + to Bus 1 and - to Ground in Simulink, save, and re-run. See matlab/instruction.md.\n');
end
fprintf('Running fourteen_bus_dynamic_probe (5 s, with probe)...\n');
sim(model);

% --- Save ScopeBus1..14 to CSV and summary ---
summaryPath = fullfile(outDir, 'summary.txt');
fid = fopen(summaryPath, 'w');
fprintf(fid, 'fourteen_bus_dynamic_probe results (with Hann probe)\n');
fprintf(fid, 'Model: %s\n', model);
fprintf(fid, 'Saved: %s\n\n', datestr(now));

for i = 1:14
    name = sprintf('ScopeBus%d', i);
    if ~exist(name, 'var'), continue; end
    d = evalin('base', name);
    t = d.time(:);
    nt = numel(t);
    fprintf(fid, '%s: time [%g, %g] s, %d points\n', name, min(t), max(t), nt);
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

% --- Derive ROCOF_max and f_min from voltage (same definition as Python) for observation ---
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
    % Clarke; phase phi = atan2(V_beta, V_alpha), unwrapped
    v_alpha = (2/3)*(va - 0.5*vb - 0.5*vc);
    v_beta = (1/sqrt(3))*(vb - vc);
    phi = atan2(v_beta, v_alpha);
    phi = unwrap(phi);
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
    % Append to summary
    fid = fopen(summaryPath, 'a');
    fprintf(fid, 'Derived (same as Python): f_nominal=%g Hz, fs=%g Hz\n', f_nominal, fs_obs);
    fprintf(fid, 'bus\tROCOF_max (Hz/s)\tf_min (Hz)\n');
    for k = 1:obsCount
        r = obsRows{k};
        fprintf(fid, '%d\t%.6f\t%.6f\n', r(1), r(2), r(3));
    end
    fclose(fid);
end

% --- ROCOF vs time plot for Bus 1 (probe bus) so probe effect is visible ---
if obsCount >= 1
    name = 'ScopeBus1';
    if exist(name, 'var')
        d = evalin('base', name);
        t = d.time(:);
        vals = d.signals(1).values;
        va = vals(:,1); vb = vals(:,2); vc = vals(:,3);
        v_alpha = (2/3)*(va - 0.5*vb - 0.5*vc);
        v_beta = (1/sqrt(3))*(vb - vc);
        phi = unwrap(atan2(v_beta, v_alpha));
        f_inst = gradient(phi, t) / (2*pi);
        delta_f = f_inst - f_nominal;
        n_obs = max(2, round((max(t)-min(t)) * fs_obs));
        t_obs = linspace(min(t), max(t), n_obs);
        delta_f_obs = interp1(t, delta_f, t_obs);
        h_obs = (max(t)-min(t)) / (n_obs - 1);
        rocof_t = gradient(delta_f_obs, h_obs);
        r1 = obsRows{1};
        rocof_max1 = r1(2);
        f = figure('Visible', 'off', 'Position', [0, 0, 900, 400], 'PaperPositionMode', 'auto');
        plot(t_obs, rocof_t, 'LineWidth', 1); hold on;
        yline(rocof_max1, '--r', sprintf('ROCOF_{max} = %.4f Hz/s', rocof_max1));
        yline(-rocof_max1, '--r');
        xlabel('Time (s)', 'FontSize', 12); ylabel('ROCOF (Hz/s)', 'FontSize', 12);
        title('Bus 1 — ROCOF from voltage (probe observable)', 'FontSize', 14);
        set(gca, 'FontSize', 11); grid on;
        saveas(f, fullfile(outDir, 'ROCOF_bus1.png'));
        close(f);
    end
end

% --- Save probe output if logged (root Outport from subsystem) ---
try
    p = evalin('base', 'yout');
    if exist('p', 'var') && isstruct(p) && isfield(p, 'signals')
        t = evalin('base', 'tout');
        M = [t(:), p.signals(1).values];
        writematrix(M, fullfile(outDir, 'ProbeOut.csv'));
    end
catch
end

% --- Plots: one per bus + overview (larger, less condensed) ---
figPos = [0, 0, 900, 500];
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
    title(sprintf('Fourteen\\_bus\\_dynamic\\_probe — Bus %d', i), 'FontSize', 14);
    set(gca, 'FontSize', 11); grid on;
    saveas(f, fullfile(outDir, [name, '.png']));
    close(f);
end

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
sgtitle('fourteen\_bus\_dynamic\_probe — all buses', 'FontSize', 14);
set(f, 'PaperPosition', [0 0 12 14]);
saveas(f, fullfile(outDir, 'all_buses.png'));
close(f);

close_system(model, 0);
fprintf('Results saved to: %s\n', outDir);

function yes = block_exists(blk)
  yes = false;
  try
    get_param(blk, 'Handle');
    yes = true;
  catch
  end
end
