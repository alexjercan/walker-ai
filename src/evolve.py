import logging
import os

import argparse
import gym.wrappers
import multiprocessing
import neat
import pickle
from dataclasses import dataclass
from pathlib import Path

import visualize


@dataclass
class Options:
    logdir: str
    config: str
    steps: int
    generations: int


def _info(opt: Options) -> None:
    try:
        int(Path(opt.logdir).parts[-1])
    except Exception:
        logging.warning(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if any(Path(opt.logdir).iterdir()):
        logging.warning("Warning! Logdir path exists, results can be corrupted.")

    logging.info(f"Saving results in {opt.logdir} using the {opt} options.")


class WalkerAgent:
    def __init__(self, opt: Options):
        self._steps = opt.steps

    def eval_genome(self, genome, config) -> float:
        env = gym.make("BipedalWalker-v3")

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        fitnesses = []

        s, _ = env.reset()
        ep_cnt = 0
        step_cnt = 0
        done = False
        episode_reward = 0

        while step_cnt < self._steps or not done:
            if done:
                fitnesses.append(episode_reward)
                ep_cnt += 1
                s, _ = env.reset()
                done = False
                episode_reward = 0

            # TODO: Maybe we want to also have random actions
            a = net.activate(s)
            s_next, r, terminated, truncated, info = env.step(a)

            done = terminated or truncated

            episode_reward += r
            s = s_next.copy()

            step_cnt += 1

        return min(fitnesses)


def main(opt: Options):
    _info(opt)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        opt.config,
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    wa = WalkerAgent(opt)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), wa.eval_genome)
    winner = pop.run(pe.evaluate, n=opt.generations)

    # Save the winner.
    with open(os.path.join(opt.logdir, "winner-feedforward"), "wb") as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(
        stats,
        ylog=True,
        view=True,
        filename=os.path.join(opt.logdir, "feedforward-fitness.svg"),
    )
    visualize.plot_species(
        stats,
        view=True,
        filename=os.path.join(opt.logdir, "feedforward-speciation.svg"),
    )

    # TODO: this
    # node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    # visualize.draw_net(config, winner, True, node_names=node_names)

    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward.gv")
    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)


def get_options() -> Options:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", dest="logdir", type=str, default="logdir/")
    parser.add_argument("--config", dest="config", type=str, default="config.ini")
    parser.add_argument(
        "--steps",
        dest="steps",
        type=int,
        metavar="STEPS",
        default=10_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "--generations",
        dest="generations",
        type=int,
        metavar="G",
        default=20,
        help="Max number of generations to evolve.",
    )

    args = parser.parse_args()

    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)

    run = str(len([f for f in os.scandir(logdir) if f.is_dir()]))
    logdir = os.path.join(logdir, run)
    os.makedirs(logdir, exist_ok=True)

    return Options(
        logdir=logdir,
        config=args.config,
        steps=args.steps,
        generations=args.generations,
    )


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
