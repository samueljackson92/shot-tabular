"""
A tabular dataset creation tool for processing signals from various data transports
and saving them in a standardized format. Supports parallel processing of shots
and flexible time alignment of signals.
"""

import argparse
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from shot_tabular.backends import registry

console = Console()


class TimeSettings(BaseModel):
    """Settings for time alignment and interpolation of signals."""

    tmin: float = 0.0
    tmax: float | None = None
    dt: float = 0.01
    method: Literal["nearest", "linear", "cubic", "zero"] = "nearest"


def process_signal(
    shot: int,
    signals: dict[str, str],
    backend: str,
    output_folder: Path,
    time_settings: TimeSettings,
) -> tuple[int, str]:
    """Process signals for a single shot and save the results to a Parquet file."""
    backend_impl = registry.get(backend)
    variables = {}
    errors = []

    for alias, signal in signals.items():
        try:
            ds = backend_impl.get(shot=shot, signal=signal)
        except Exception as e:
            errors.append(f"Error loading signal {signal}: {e}")
            continue

        if "data" not in ds or "time" not in ds:
            errors.append(f"Signal {signal} is missing 'data' or 'time' variables")
            continue

        ds = ds["data"]
        tmax = (
            time_settings.tmax
            if time_settings.tmax is not None
            else ds["time"].max().item()
        )
        time_base = np.arange(time_settings.tmin, tmax, time_settings.dt)
        variable = ds.interp(time=time_base, method=time_settings.method)
        variables[alias] = pd.Series(variable.values, name=alias, index=time_base)

    if len(variables) == 0:
        error_msg = f"No valid signals found for shot {shot}"
        if errors:
            error_msg += f": {'; '.join(errors)}"
        return shot, error_msg

    df = pd.DataFrame(variables)
    cols = df.columns.tolist()

    for alias in signals.keys():
        if alias not in cols:
            df[alias] = np.nan  # Add missing signal as NaN column
            df[alias] = df[alias].astype("float64")

    df["time"] = np.arange(len(df)) * time_settings.dt + time_settings.tmin
    df["shot"] = shot
    df.index.name = "index"
    df = df[["shot", "time"] + list(signals.keys())]  # Reorder columns
    df = df.sort_values(["shot", "time"])  # Sort by shot and time
    df = df.set_index("time")

    schema = pa.schema(
        [
            pa.field("shot", pa.int32()),
            pa.field("time", pa.float64()),
        ]
        + [pa.field(alias, pa.float64()) for alias in signals.keys()]
    )

    try:
        table = pa.Table.from_pandas(df, schema=schema)
        pq.write_table(table, output_folder / f"{shot}.parquet")
        if errors:
            return shot, f"Partial success with warnings: {'; '.join(errors)}"
        return shot, ""
    except Exception as e:
        return shot, f"Error saving parquet file: {e}"


class StoreKeyValue(argparse.Action):
    """Custom argparse action to store key-value pairs in a dictionary"""

    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        key = values[0]
        value = values[1] if len(values) > 1 else key
        getattr(namespace, self.dest)[key] = value


def gather_results(jobs, total_shots):
    """Gather results from parallel processing and display progress."""
    success_count = 0
    error_count = 0
    warning_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing shots...", total=total_shots)

        for shot, message in jobs:
            if message == "":
                console.print(f"[green]✅ Successfully processed shot {shot}[/green]")
                success_count += 1
            elif "warning" in message.lower() or "partial" in message.lower():
                console.print(f"[yellow]⚠️  Shot {shot}: {message}[/yellow]")
                warning_count += 1
            else:
                console.print(f"[red]❌ Shot {shot}: {message}[/red]")
                error_count += 1

            progress.advance(task)

    # Create summary table
    table = Table(
        title="Processing Summary", show_header=False, box=None, padding=(0, 1)
    )
    table.add_column(justify="left", no_wrap=True)  # Emoji column
    table.add_column(justify="left", no_wrap=True)  # Label column
    table.add_column(justify="right", no_wrap=True)  # Number column

    table.add_row(
        "[green]✅[/green]",
        "[green]Successful[/green]",
        f"[bold]{success_count:>6}[/bold]",
    )
    table.add_row(
        "[yellow]⚠️[/yellow]",
        "[yellow]Warnings[/yellow]",
        f"[bold]{warning_count:>6}[/bold]",
    )
    table.add_row(
        "[red]❌[/red]",
        "[red]Errors[/red]",
        f"[bold]{error_count:>6}[/bold]",
    )
    table.add_row(
        "[blue]📊[/blue]",
        "[blue]Total[/blue]",
        f"[bold]{total_shots:>6}[/bold]",
    )

    console.print()
    console.print(table)
    console.print()

    return success_count, error_count, warning_count


def main():
    """Main function to parse arguments and orchestrate the processing of shots."""
    parser = argparse.ArgumentParser(description="A tabular dataset creation tool")
    parser.add_argument("--shots", type=int, nargs="+", help="Shots to process")
    parser.add_argument(
        "--shot-file", type=str, help="File containing list of shots to process"
    )
    parser.add_argument("--shot-min", type=int, help="Minimum shot number")
    parser.add_argument("--shot-max", type=int, help="Maximum shot number")
    parser.add_argument(
        "-s", dest="signals", action=StoreKeyValue, nargs="+", metavar=("KEY", "VALUE")
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["uda", "sal"],
        default="uda",
        help="Data backend method",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        default="output",
        help="Output folder path",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes to use",
    )
    parser.add_argument(
        "--tmin", type=float, help="Minimum time to process", default=0.0
    )
    parser.add_argument(
        "--tmax", type=float, help="Maximum time to process", default=None
    )
    parser.add_argument(
        "--dt", type=float, help="Time step for processing", default=0.01
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["nearest", "linear", "cubic", "zero"],
        default="linear",
        help="Interpolation method for time alignment",
    )
    args = parser.parse_args()

    if args.shots:
        shots = args.shots
    elif args.shot_file:
        shot_file_path = Path(args.shot_file).expanduser().resolve()
        if shot_file_path.suffix == ".csv":
            shots = pd.read_csv(shot_file_path).iloc[:, 0].astype(int).tolist()
        elif shot_file_path.suffix == ".parquet":
            shots = pd.read_parquet(shot_file_path).index.values.astype(int).tolist()
        else:
            console.print(
                "[red]❌ Error: Unsupported shot file format. "
                "Use .csv or .parquet[/red]"
            )
            sys.exit(1)
    elif args.shot_min is not None and args.shot_max is not None:
        shots = list(range(args.shot_min, args.shot_max + 1))
    else:
        console.print(
            "[red]❌ Error: You must specify either --shots, --shot-file, "
            "or both --shot-min and --shot-max[/red]"
        )
        sys.exit(1)

    output_folder = Path(args.output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    signals = args.signals
    time_settings = TimeSettings(
        tmin=args.tmin, tmax=args.tmax, dt=args.dt, method=args.method
    )

    # Print startup information
    console.print()
    console.print("[bold cyan]🚀 Starting Shot Tabular Processing[/bold cyan]")
    console.print(f"[dim]Shots to process: {len(shots)}[/dim]")
    console.print(f"[dim]Workers: {args.num_workers}[/dim]")
    console.print(f"[dim]Backend: {args.backend}[/dim]")
    console.print(f"[dim]Output directory: {output_folder}[/dim]")
    console.print()

    with mp.Pool(args.num_workers) as pool:
        jobs = pool.imap_unordered(
            partial(
                process_signal,
                signals=signals,
                backend=args.backend,
                output_folder=output_folder,
                time_settings=time_settings,
            ),
            shots,
        )

        success_count, error_count, warning_count = gather_results(jobs, len(shots))

    # Final summary
    if error_count > 0 or warning_count > 0:
        console.print(
            f"[yellow]⚠️  Processing completed with {error_count} errors "
            f"and {warning_count} warnings.[/yellow]"
        )
    else:
        console.print(
            f"[green bold]✅ Processing completed successfully! "
            f"{success_count} shots processed.[/green bold]"
        )

    console.print()
    console.print(f"[green]💾 Results saved to:[/green] [bold]{output_folder}[/bold]")
    console.print("[dim]   Format: Parquet[/dim]")
    console.print(f"[dim]   Files: {success_count}[/dim]")
    console.print()


if __name__ == "__main__":
    main()
