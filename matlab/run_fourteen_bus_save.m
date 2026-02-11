%% Run Fourteen_bus (0.12 s steady-state) and save all results to CSV, TXT, PNG
% Usage: run('run_fourteen_bus_save.m')
% Outputs: results/fourteen_bus/*.csv, *.txt, *.png

scriptDir = fileparts(mfilename('fullpath'));
if ~isempty(scriptDir), cd(scriptDir); end

outDir = fullfile(scriptDir, 'results', 'fourteen_bus');
if ~isfolder(outDir), mkdir(outDir); end

model = 'Fourteen_bus.mdl';
fprintf('Running Fourteen_bus (steady-state 0.12 s)...\n');
load_system(model);
set_param('Fourteen_bus', 'UnconnectedOutputMsg', 'none');
sim('Fourteen_bus');

% --- Save tout, xout, yout to CSV ---
if exist('tout', 'var')
    writematrix(tout(:), fullfile(outDir, 'tout.csv'));
end
if exist('xout', 'var')
    if isstruct(xout) && isfield(xout, 'signals')
        V = xout.signals(1).values;
        if size(V, 1) == numel(tout), M = [tout(:), V]; else, M = V; end
        writematrix(M, fullfile(outDir, 'xout.csv'));
    elseif isstruct(xout) && isfield(xout, 'values')
        writematrix(xout.values, fullfile(outDir, 'xout.csv'));
    else
        writematrix(xout, fullfile(outDir, 'xout.csv'));
    end
end
if exist('yout', 'var')
    if isstruct(yout) && isfield(yout, 'signals')
        V = yout.signals(1).values;
        if size(V, 1) == numel(tout), M = [tout(:), V]; else, M = V; end
        writematrix(M, fullfile(outDir, 'yout.csv'));
    elseif isstruct(yout) && isfield(yout, 'values')
        writematrix(yout.values, fullfile(outDir, 'yout.csv'));
    else
        writematrix(yout, fullfile(outDir, 'yout.csv'));
    end
end

% --- Summary TXT ---
fid = fopen(fullfile(outDir, 'summary.txt'), 'w');
fprintf(fid, 'Fourteen_bus (steady-state) results\n');
fprintf(fid, 'Model: %s\n', model);
fprintf(fid, 'Saved: %s\n\n', datestr(now));
if exist('tout', 'var')
    fprintf(fid, 'tout: %d points, range [%g, %g] s\n', numel(tout), min(tout), max(tout));
end
if exist('xout', 'var')
    fprintf(fid, 'xout: saved to xout.csv\n');
end
if exist('yout', 'var')
    fprintf(fid, 'yout: saved to yout.csv\n');
end
fclose(fid);

% --- Plots (PNG) ---
if exist('tout', 'var') && exist('xout', 'var')
    f = figure('Visible', 'off');
    if isstruct(xout) && isfield(xout, 'signals')
        plot(tout, xout.signals(1).values);
    elseif isstruct(xout) && isfield(xout, 'values')
        plot(tout, xout.values);
    else
        plot(tout, xout);
    end
    xlabel('Time (s)'); ylabel('State'); title('Fourteen\_bus: state vs time');
    saveas(f, fullfile(outDir, 'xout_vs_time.png'));
    close(f);
end
if exist('tout', 'var') && exist('yout', 'var')
    f = figure('Visible', 'off');
    if isstruct(yout) && isfield(yout, 'signals')
        plot(tout, yout.signals(1).values);
    elseif isstruct(yout) && isfield(yout, 'values')
        plot(tout, yout.values);
    else
        plot(tout, yout);
    end
    xlabel('Time (s)'); ylabel('Output'); title('Fourteen\_bus: output vs time');
    saveas(f, fullfile(outDir, 'yout_vs_time.png'));
    close(f);
end

fprintf('Results saved to: %s\n', outDir);
