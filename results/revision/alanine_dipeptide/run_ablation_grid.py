#!/usr/bin/env python3
"""Plan or run the alanine window/trajectory grid from existing trajectories.

Default: write a 60-cell manifest only. Pass --run to execute sequentially.
Fractions retain contiguous trajectory prefixes, with no extra equilibration cut.
Each completed cell has a configuration-checked marker; failures stop the run.
Use --window-counts 18 --fractions 0.04 --run for a small timing pilot.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import pty
import shutil
import struct
import subprocess
import sys
import tempfile
import termios
import time

HERE = Path(__file__).resolve().parent
WINDOWS = {18: 'xvg_data_32', 36: 'xvg_data_16', 72: 'xvg_data_8',
           144: 'xvg_data_quarter', 288: 'xvg_data_half', 576: 'xvg_data'}
FRACTIONS = [1., .9, .8, .6, .4, .25, .16, .1, .063, .04]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--window-counts', nargs='+', type=int, choices=list(WINDOWS), default=list(WINDOWS))
    parser.add_argument('--fractions', nargs='+', type=float, default=FRACTIONS)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'data_ablation_results' / 'trajectory_grid')
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--num-samples', type=int, default=250)
    parser.add_argument('--num-chains', type=int, default=4)
    parser.add_argument('--max-tree-depth', type=int, default=4,
                        help='NUTS max tree depth; lower values cap leapfrog steps per iteration.')
    parser.add_argument('--skip-hmc', action='store_true')
    args = parser.parse_args()
    if any(not 0 < f <= 1 for f in args.fractions):
        parser.error('Fractions must lie in (0, 1].')
    if min(args.warmup_steps, args.num_samples, args.num_chains) < 1:
        parser.error('Sampling counts must be positive.')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for count in args.window_counts:
        source = HERE / 'umbrella_data' / WINDOWS[count]
        files = sorted(source.glob('*_xyplane.xvg'))
        if len(files) != count or not (source / 'README').exists():
            raise ValueError(f'Invalid dataset: {source}')
        for fraction in args.fractions:
            cells.append(dict(windows=count, fraction=fraction, source=str(source),
                              output=str(args.output_dir / f'{count}_windows' / f'fraction_{fraction:g}'),
                              warmup_steps=args.warmup_steps, num_samples=args.num_samples,
                              num_chains=args.num_chains, max_tree_depth=args.max_tree_depth,
                              skip_hmc=args.skip_hmc))
    # Selected pilot plans do not overwrite the full-grid manifest.
    plan = args.output_dir / ('run_plan.json' if args.run else 'grid_manifest.json')
    plan.write_text(json.dumps(cells, indent=2) + '\n')
    print(f'{len(cells)} cells; plan: {plan}', flush=True)
    if not args.run:
        return
    for cell in cells:
        out = Path(cell['output'])
        marker = out / 'completed.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved['configuration'] != cell:
                raise ValueError(f'Configuration changed for {out}; choose another output directory.')
            if (out / 'synthetic_2D_reconstruction.csv').exists():
                print(f'Skip completed {out}', flush=True)
                continue
        out.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='alanine-ablation-') as temporary:
            staged = Path(temporary)
            source = Path(cell['source'])
            shutil.copy2(source / 'README', staged / 'README')
            for path in sorted(source.glob('*_xyplane.xvg')):
                lines = path.read_text().splitlines(keepends=True)
                headers = [line for line in lines if line.lstrip().startswith(('#', '@'))]
                samples = [line for line in lines if line.strip() and not line.lstrip().startswith(('#', '@'))]
                keep = max(2, int(len(samples) * cell['fraction']))
                (staged / path.name).write_text(''.join(headers + samples[:keep]))
            command = [sys.executable, str(HERE / 'run_2D_reconstruction.py'),
                       '--dataset-root', str(staged), '--results-dir', str(out),
                       '--reference-path', str(HERE / 'umbrella_data/xvg_data/wham_reference.csv'),
                       '--warmup-steps', str(args.warmup_steps), '--num-samples', str(args.num_samples),
                       '--num-chains', str(args.num_chains),
                       '--max-tree-depth', str(args.max_tree_depth)]
            if args.skip_hmc:
                command.append('--skip-hmc')
            (out / 'configuration.json').write_text(json.dumps(cell, indent=2) + '\n')
            print(f'Running {cell["windows"]} windows, fraction {cell["fraction"]}', flush=True)
            # Stream child output live to the terminal while teeing to run.log.
            # A PTY is used so tqdm/Pyro progress bars (which only render on a TTY)
            # display correctly; control codes are stripped before logging.
            master_fd, slave_fd = pty.openpty()
            # A fresh PTY has a 0x0 window size, which makes tqdm render empty
            # progress bars (bare carriage returns). Propagate the parent
            # terminal's size so bars render normally.
            try:
                parent_cols, parent_rows = os.get_terminal_size()
            except OSError:
                parent_cols, parent_rows = 120, 50
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ,
                        struct.pack('HHHH', max(parent_rows, 24), max(parent_cols, 80), 0, 0))
            # Disable ONLCR (\n -> \r\n) translation on the PTY: the extra
            # carriage returns would break tqdm's cursor-up positioning, making
            # multi-bar redraws drift down one line per update instead of
            # overwriting the existing bars in place.
            attrs = termios.tcgetattr(slave_fd)
            attrs[1] = attrs[1] & ~termios.OPOST
            termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
            with (out / 'run.log').open('w') as log:
                process = subprocess.Popen(command, stdout=slave_fd, stderr=slave_fd,
                                           env={**os.environ, 'MPLBACKEND': 'Agg'})
                os.close(slave_fd)
                while True:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                    # Collapse carriage-return progress redraws into newline logs.
                    text = chunk.decode('utf-8', errors='replace').replace('\r', '\n')
                    log.write(text)
                    log.flush()
                os.close(master_fd)
                returncode = process.wait()
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, command)
        marker.write_text(json.dumps(dict(configuration=cell, elapsed_seconds=time.monotonic()-start), indent=2)+'\n')
        print(f'Completed in {time.monotonic()-start:.1f} s', flush=True)


if __name__ == '__main__':
    main()
