import logging
import os

import argparse
import cv2
import glob
import gym
import neat
import numpy as np
from dataclasses import dataclass
from gym.envs.registration import register
from neat.six_util import itervalues
from pathlib import Path
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
from tqdm.auto import tqdm

import visualize
from pygame_recorder import ScreenRecorder

register(
    id="BipedalWalker-v4",
    entry_point="bipedal_walker:BipedalWalker",
    max_episode_steps=1600,
    reward_threshold=300,
)


@dataclass
class Options:
    """the options of the visualization simulation

    Attributes
    ==========
    logdir : str
        path to load the results from
    config : str
        path to the config.ini file with the definitions of the genome
    steps : int
        maximum number of steps to run the evaluation of the genome on the environmnet
    hardcore : bool
        play the game in hardcore
    """

    logdir: str
    config: str
    steps: int
    hardcore: bool


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


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def main(opt: Options):
    _info(opt)

    VIEWPORT_H = 1080
    VIEWPORT_W = 1920
    FPS = 50

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
    checkpoint_paths = sorted(checkpoint_paths, key=lambda n: int(n.split("-")[-1]))

    env = gym.make("BipedalWalker-v4", render_mode="human", hardcore=opt.hardcore)

    visualization_dir = "visualization" + ("_hardcore" if opt.hardcore else "_normal")
    os.makedirs(os.path.join(opt.logdir, visualization_dir), exist_ok=True)

    recorder = ScreenRecorder(
        VIEWPORT_W,
        VIEWPORT_H,
        FPS,
        out_file=os.path.join(opt.logdir, visualization_dir, "output.mp4"),
    )

    pbar = tqdm(total=len(checkpoint_paths), position=0, leave=True)
    for checkpoint_path in checkpoint_paths:
        name = Path(checkpoint_path).name
        generation = name.split("-")[-1]

        logging.info(f"Using {name} population")

        pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)

        winner = None
        for g in itervalues(pop.population):
            if winner is None or (g.fitness is not None and g.fitness > winner.fitness):
                winner = g

        node_names = {0: "hip_1", 1: "knee_1", 2: "hip_2", 3: "knee_2"}
        visualize.draw_net(
            config,
            winner,
            view=False,
            node_names=node_names,
            filename=os.path.join(
                opt.logdir, visualization_dir, f"{name}-feedforward.gv"
            ),
        )

        net_img = svg2rlg(
            os.path.join(opt.logdir, visualization_dir, f"{name}-feedforward.gv.svg")
        )
        net_img = renderPM.drawToPIL(net_img, dpi=144)
        net_img = np.array(net_img)[:, :, ::-1]

        net_img = image_resize(net_img, width=VIEWPORT_W * 3 // 4)

        image_extended = np.ones((VIEWPORT_H, VIEWPORT_W, 3), dtype=net_img.dtype) * 255
        image_extended[: net_img.shape[0], -net_img.shape[1] :] = net_img[
            :VIEWPORT_H, :VIEWPORT_W, :
        ]

        net_img = image_extended

        net = neat.nn.FeedForwardNetwork.create(winner, config)

        s, _ = env.reset()
        step_cnt = 0
        done = False
        episode_reward = 0

        try:
            while step_cnt < opt.steps and not done:
                env.render()
                recorder.capture_frame(
                    env.screen,
                    text=f"Generation: {generation}\nReward: {episode_reward:.02f}",
                    overlay=net_img,
                )

                a = net.activate(s)
                s_next, r, terminated, truncated, info = env.step(a)

                done = terminated or truncated

                episode_reward += r
                s = s_next.copy()

                step_cnt += 1

        except KeyboardInterrupt:
            pass

        pbar.update(1)
    recorder.end_recording()


def get_options() -> Options:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", dest="logdir", type=str, default="logdir/0")
    parser.add_argument("--config", dest="config", type=str, default="config.ini")
    parser.add_argument(
        "--steps",
        dest="steps",
        type=int,
        metavar="STEPS",
        default=1_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "--hardcore", dest="hardcore", action="store_true", default=False
    )

    args = parser.parse_args()

    logdir = args.logdir

    return Options(
        logdir=logdir,
        config=args.config,
        steps=args.steps,
        hardcore=args.hardcore,
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
