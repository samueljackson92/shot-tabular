# shot-tabular

`shot-tabular` is a command-line tool for building tabular datasets from shot-based time series signals.

It loads one or more signals for each shot, interpolates them onto a shared time base, and writes a single combined table to CSV or Parquet.

## Features

- Load multiple signals per shot using `uda` or `sal` transport
- Interpolate each signal to a common time grid
- Process many shots in parallel with multiprocessing
- Export the merged result to CSV (`.csv`) or Parquet (`.parquet`)

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
	--output-file out.csv
```

### Shot Range Example

```bash
st \
	--shot-min 12000 \
	--shot-max 12010 \
	-s ip \
	-o out.parquet
```

## Output Format

The output includes:

- `shot`: shot number
- `time`: generated time base from `tmin`, `tmax`, and `dt`
- one column per signal alias

Rows from all shots are concatenated into a single table.

## Main Options

- `--shots`: explicit list of shot numbers
- `--shot-file`: CSV/text file of shot numbers (first column)
- `--shot-min`, `--shot-max`: inclusive shot range
- `-s`: signal mapping(s), repeatable
- `--transport`: `uda` or `sal` (default: `uda`)
- `-o`, `--output-file`: output path (default: `out.csv`)
- `-n`, `--num-workers`: worker process count (default: CPU count)
- `--tmin`: start time (default: `0.0`)
- `--tmax`: end time (default: signal max time)
- `--dt`: time step (default: `0.01`)
- `--method`: interpolation method (`nearest`, `linear`, `cubic`, `zero`, `next`, `previous`)

## Notes

- If an individual signal fails to load for a shot, processing continues and the error is printed.
- CSV is used unless `--output-file` ends with `.parquet`.
