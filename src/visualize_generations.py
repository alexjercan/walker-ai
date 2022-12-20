import gym
import logging
import os

import argparse
import glob
import neat
from dataclasses import dataclass
from neat.six_util import itervalues
from pathlib import Path

import visualize
from pygame_recorder import ScreenRecorder
from gym.envs.box2d.bipedal_walker import VIEWPORT_H, VIEWPORT_W, FPS


@dataclass
class Options:
    """the options of the visualization simulation

    Attributes
    ==========
    logdir : str
        path to load the results from
    config : str
        path to the config.ini file with the definitions of the genome
    """

    logdir: str
    config: str


def _info(opt: Options) -> None:
    try:
        int(Path(opt.logdir).parts[-1])
    except Exception:
        logging.warning(
            "Warning, logdir path should end in a number indicating a training run."
        )
    if not any(Path(opt.logdir).iterdir()):
        logging.error(
            "Warning! Logdir does not path exists. Make sure to use an existing logdir."
        )
        exit()

    logging.info(f"Loading results from {opt.logdir} using the {opt} options.")


def main(opt: Options):
    _info(opt)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        opt.config,
    )

    checkpoint_paths = glob.glob(
        os.path.join(opt.logdir, "checkpoint", "neat-checkpoint-*")
    )

    env = gym.make("BipedalWalker-v3", render_mode="human")
    recorder = ScreenRecorder(VIEWPORT_W, VIEWPORT_H, FPS, out_file=os.path.join(opt.logdir, "visualization", "output.avi"))

    for checkpoint_path in checkpoint_paths:
        name = Path(checkpoint_path).name

        logging.info(f"Using {name} population")

        pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)

        winner = None
        for g in itervalues(pop.population):
            if winner is None or (g.fitness is not None and g.fitness > winner.fitness):
                winner = g

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
            filename=os.path.join(opt.logdir, "visualization", f"{name}-feedforward.gv"),
        )
        visualize.draw_net(
            config,
            winner,
            view=False,
            node_names=node_names,
            filename=os.path.join(opt.logdir, "visualization", f"{name}-feedforward-enabled-pruned.gv"),
            prune_unused=True,
        )

        net = neat.nn.FeedForwardNetwork.create(winner, config)

        s, _ = env.reset()
        done = False

        try:
            while not done:
                env.render()
                recorder.capture_frame(env.screen)

                a = net.activate(s)
                s_next, r, terminated, truncated, info = env.step(a)

                done = terminated or truncated

                s = s_next.copy()

        except KeyboardInterrupt:
            continue

    recorder.end_recording()


def get_options() -> Options:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", dest="logdir", type=str, default="logdir/0")
    parser.add_argument("--config", dest="config", type=str, default="config.ini")

    args = parser.parse_args()

    logdir = args.logdir

    os.makedirs(os.path.join(logdir, "visualization"), exist_ok=True)

    return Options(
        logdir=logdir,
        config=args.config,
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
