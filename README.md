# shot-tabular

`shot-tabular` is a command-line tool for building per-shot tabular datasets from shot-based time series signals.

It loads one or more signals for each shot, interpolates them onto a shared time base, and writes one Parquet file per shot.

## Features

- Load multiple signals per shot using `uda` or `sal` transport
- Interpolate each signal to a common time grid
- Process many shots in parallel with multiprocessing
- Save one output file per shot as Parquet (`<output-dir>/<shot>.parquet`)

## Installation

This project uses `pyproject.toml` and requires Python 3.12+.

If you use `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI

After installation, the command is:

```bash
st
```

You must specify shots in one of these ways:

- `--shots 12345 12346`
- `--shot-file shots.csv` (first column is used)
- `--shot-min 12345 --shot-max 12360`

You should also provide one or more `-s` signal mappings:

- `-s <alias> <signal>` to map a column name to a signal path
- `-s <signal>` to use the same text for alias and signal

### Basic Example

```bash
st \
	--shots 12345 12346 \
	-s ip "AMC_PLASMA CURRENT" \
	-s ne "ANE_DENSITY" \
	--transport uda \
	--tmin 0.0 \
	--tmax 5.0 \
	--dt 0.01 \
	--method nearest \
	--output-file output
```

### Shot Range Example

```bash
st \
	--shot-min 12000 \
	--shot-max 12010 \
	-s ip \
	-o output
```

## Output Format

Each successfully processed shot is written to:

- `<output-dir>/<shot>.parquet`

Each per-shot table includes:

- `shot`: shot number
- `time`: generated time base from `tmin`, `tmax`, and `dt`
- one column per signal alias

Columns are ordered as `shot`, `time`, then the successfully loaded signal columns.

## Main Options

- `--shots`: explicit list of shot numbers
- `--shot-file`: CSV/text file of shot numbers (first column)
- `--shot-min`, `--shot-max`: inclusive shot range
- `-s`: signal mapping(s), repeatable
- `--transport`: `uda` or `sal` (default: `uda`)
- `-o`, `--output-file`: output directory path (default: `output`)
- `-n`, `--num-workers`: worker process count (default: CPU count)
- `--tmin`: start time (default: `0.0`)
- `--tmax`: end time (default: signal max time)
- `--dt`: time step (default: `0.01`)
- `--method`: interpolation method (`nearest`, `linear`, `cubic`, `zero`, `next`, `previous`)

## Notes

- If an individual signal fails to load for a shot, processing continues and the error is printed.
- Output is currently Parquet-only.
- If all signals fail for a shot, no file is written for that shot.
