import logging
import os

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Options:
    logdir: str
    agent: str


def _info(opt: Options) -> None:
    logger = logging.getLogger()

    try:
        int(Path(opt.logdir).parts[-1])
    except Exception:
        logger.warning(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if any(Path(opt.logdir).iterdir()):
        logger.warning("Warning! Logdir path exists, results can be corrupted.")

    logger.info(f"Saving results in {opt.logdir} using the {opt.agent} agent.")


def main(opt: Options) -> None:
    _info(opt)


def get_options() -> Options:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", dest="logdir", type=str, default="logdir/")
    parser.add_argument(
        "--agent",
        dest="agent",
        type=str,
        default="random_agent",
        help="The agent to use",
    )

    args = parser.parse_args()

    logdir = os.path.join(args.logdir, f"{args.agent}_agent")
    os.makedirs(logdir, exist_ok=True)

    run = str(len([f for f in os.scandir(logdir) if f.is_dir()]))
    logdir = os.path.join(logdir, run)
    os.makedirs(logdir, exist_ok=True)

    return Options(logdir=logdir, agent=args.agent)


if __name__ == "__main__":
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.getenv("LOG_FILE", "train.log")),
        ],
        level=logging.DEBUG,
        format="%(levelname)s: %(asctime)s \
            pid:%(process)s module:%(module)s %(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
    )

    main(get_options())
