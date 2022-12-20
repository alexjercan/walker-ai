import itertools
import logging
import os

import argparse
import gym.wrappers
import multiprocessing
import neat
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import visualize


@dataclass
class Options:
    """the options of the evolution simulation

    Attributes
    ==========
    logdir : str
        path to save the results to
    config : str
        path to the config.ini file with the definitions of the genome
    steps : int
        maximum number of steps to run the evaluation of the genome on the environmnet
    generations : int
        maximum number of generations to evolve
    """

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


class EvalGenomeBuilder:
    def __init__(self, opt: Options):
        """constructs the genome builder object

        Parameters
        ==========
        opt : Options
            used to specify the options of this genome builder
        """
        self._steps = opt.steps

    def eval_genome(self, genome, config) -> float:
        """evaluate the fitness of a genome

        Parameters
        ==========
        genome : DefaultGenome
            the genome that is being evaluated
        config: Config
            the configuration of the NEAT algorithm

        Returns
        =======
        float
            the worst fitness of this genome
        """

        # initialize the env
        env = gym.make("BipedalWalker-v3")
        # initialize the phenotype (feed forward network)
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        s, _ = env.reset()
        ep_cnt = 0
        step_cnt = 0
        done = False
        episode_reward = 0

        fitnesses = []
        while step_cnt < self._steps or not done:
            # if the agent finishes the run then append the reward as the fitness level
            if done:
                fitnesses.append(episode_reward)
                ep_cnt += 1
                s, _ = env.reset()
                done = False
                episode_reward = 0

            # choose the next state using a greedy selection
            a = net.activate(s)
            s_next, r, terminated, truncated, info = env.step(a)

            done = terminated or truncated

            episode_reward += r
            s = s_next.copy()

            step_cnt += 1

        # return the worst fitness
        return min(fitnesses)


def main(opt: Options):
    _info(opt)

    # initialize the config using the config.ini file
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        opt.config,
    )

    # create a population
    pop = neat.Population(config)

    # create a stats reporter for nice logging
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    os.makedirs(os.path.join(opt.logdir, "checkpoint"), exist_ok=True)
    pop.add_reporter(
        neat.Checkpointer(
            1,
            900,
            filename_prefix=os.path.join(opt.logdir, "checkpoint", "neat-checkpoint-"),
        )
    )

    # evolve the agents
    eg = EvalGenomeBuilder(opt)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eg.eval_genome)
    winner = pop.run(pe.evaluate, n=opt.generations)

    # Save the winner.
    with open(os.path.join(opt.logdir, "winner-feedforward"), "wb") as f:
        pickle.dump(winner, f)

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

    node_names = (
        {
            -1: "hull_angle",
            -2: "hull_angularVelocity",
            -3: "vel_x",
            -4: "vel_y",
            -5: "hip_joint_1_angle",
            -6: "hip_joint_1_speed",
            -7: "knee_joint_1_angle",
            -8: "knee_joint_1_speed",
            -9: "leg_1_ground_contact_flag",
            -10: "hip_joint_2_angle",
            -11: "hip_joint_2_speed",
            -12: "knee_joint_2_angle",
            -13: "knee_joint_2_speed",
            -14: "leg_2_ground_contact_flag",
        }
        | {(-i - 15): f"lidar{i}" for i in range(10)}
        | {0: "hip_1", 1: "knee_1", 2: "hip_2", 3: "knee_2"}
    )
    visualize.draw_net(
        config,
        winner,
        view=False,
        node_names=node_names,
        filename=os.path.join(opt.logdir, "winner-feedforward.gv"),
    )
    visualize.draw_net(
        config,
        winner,
        view=False,
        node_names=node_names,
        filename=os.path.join(opt.logdir, "winner-feedforward-enabled-pruned.gv"),
        prune_unused=True,
    )


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
