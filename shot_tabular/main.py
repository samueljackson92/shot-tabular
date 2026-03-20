import sys
import argparse
import multiprocessing as mp
from typing import Literal
from functools import partial

import xarray as xr
import pandas as pd
import numpy as np
from pydantic import BaseModel
from loguru import logger


class TimeSettings(BaseModel):
    tmin: float = 0.0
    tmax: float | None = None
    dt: float = 0.01
    method: Literal["nearest", "linear", "cubic", "zero", "next", "previous"] = (
        "nearest"
    )


def process_signal(
    shot: int, signals: dict[str, str], transport: str, time_settings: TimeSettings
) -> pd.DataFrame | str:
    variables = {}
    for alias, signal in signals.items():
        try:
            ds = xr.open_dataset(f"{transport}://{signal}:{shot}")
        except Exception as e:
            logger.error(f"Error loading signal {signal} for shot {shot}: {e}")
            continue

        if "data" not in ds or "time" not in ds:
            logger.error(
                f"Signal {signal} for shot {shot} is missing 'data' or 'time' variables"
            )
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
        return f"No valid signals found for shot {shot}"

    df = pd.DataFrame(variables)
    df["time"] = np.arange(len(df)) * time_settings.dt + time_settings.tmin
    df["shot"] = shot
    df.index.name = "index"
    df = df[["shot", "time"] + list(signals.keys())]  # Reorder columns
    return df


class StoreKeyValue(argparse.Action):
    """Custom argparse action to store key-value pairs in a dictionary"""

    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        key = values[0]
        value = values[1] if len(values) > 1 else key
        getattr(namespace, self.dest)[key] = value


def main():
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
        "--transport",
        type=str,
        choices=["uda", "sal"],
        default="uda",
        help="Data transport method",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="out.csv",
        help="Output file path",
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
        choices=["nearest", "linear", "cubic", "zero", "next", "previous"],
        default="nearest",
        help="Interpolation method for time alignment",
    )
    args = parser.parse_args()

    if args.shots:
        shots = args.shots
    elif args.shot_file:
        shots = pd.read_csv(args.shot_file, header=None).iloc[:, 0].tolist()
    elif args.shot_min is not None and args.shot_max is not None:
        shots = list(range(args.shot_min, args.shot_max + 1))
    else:
        logger.error(
            "Error: You must specify either --shots, --shot-file, or both --shot-min and --shot-max"
        )
        sys.exit(1)

    signals = args.signals
    time_settings = TimeSettings(
        tmin=args.tmin, tmax=args.tmax, dt=args.dt, method=args.method
    )

    results = []
    with mp.Pool(args.num_workers) as pool:
        jobs = pool.imap_unordered(
            partial(
                process_signal,
                signals=signals,
                transport=args.transport,
                time_settings=time_settings,
            ),
            shots,
        )

        for result in jobs:
            if isinstance(result, str):
                logger.error(result)
            else:
                logger.info(f"Processed shot {result['shot'].iloc[0]}")
                results.append(result)

    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.sort_values(["shot", "time"])
    if args.output_file.endswith(".parquet"):
        final_df.to_parquet(args.output_file)
    else:
        final_df.to_csv(args.output_file)

    logger.info(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
