from typing import List, Tuple, Optional
from NEATObjects.EvalFunc import FitnessEvaluator
from NEATObjects.NEAT import NEATTrainer
from GameObjects.Snake import SnakeGame
from neat.checkpoint import Checkpointer

import neat
import json
import configparser
import os


class Experiment:
    def __init__(
        self,
        config_path: str,
        output_dir: str = '.',
        generations: int = 50
    ):
        self.config_path = config_path
        self.generations = generations

        # Setup experiment directory
        cfg_name = os.path.splitext(os.path.basename(config_path))[0]
        self.exp_dir = os.path.join(output_dir, cfg_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Paths for genome and state files
        self.genome_path = os.path.join(self.exp_dir, 'best_genome.pkl')
        self.states_path = os.path.join(self.exp_dir, 'game_states.json')

        # Parse config
        parser = configparser.ConfigParser()
        if not parser.read(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        # Game section
        if 'GAME' not in parser:
            raise KeyError("Missing [GAME] section in config")
        game_cfg = parser['GAME']
        gw = game_cfg.getint('grid_width', 30)
        gh = game_cfg.getint('grid_height', 30)
        cs = game_cfg.getint('cell_size', 20)
        gm = game_cfg.getint('game_mode', 1)
        # Evaluator section
        eval_name = parser.get('EVALUATOR', 'name', fallback='balanced')
        if eval_name not in EVALUATORS:
            raise KeyError(f"Unknown evaluator: {eval_name}")
        self.evaluator = EVALUATORS[eval_name]()

        # Create game and trainer
        self.game_train = SnakeGame(gw, gh, cs, gm)
        self.game_play  = SnakeGame(gw, gh, cs, gm)
        self.trainer = NEATTrainer(
            config_path=self.config_path,
            game_train=self.game_train,
            game_play=self.game_play,
            evaluator=self.evaluator
        )
        # Redirect NEAT checkpoints
        from neat.reporting import ReporterSet
        rs = self.trainer.pop.reporters
        # remove old
        old = [r for r in rs.reporters if isinstance(r, Checkpointer)]
        for r in old:
            rs.remove(r)
        # add new
        prefix = os.path.join(self.exp_dir, 'neat-checkpoint-')
        self.trainer.pop.add_reporter(Checkpointer(generation_interval=10, filename_prefix=prefix))

        # Results placeholders
        self.best_genome: Optional[object] = None
        self.score: Optional[float] = None
        self.states: Optional[list] = None

    def run(self) -> float:
        self.best_genome = self.trainer.learn(self.generations)
        self.trainer.save_genome(self.best_genome, self.genome_path)
        # Evaluate and record states
        self.score = self.trainer.play(self.best_genome, render=False, states_path=self.states_path)
        # Load states
        with open(self.states_path) as f:
            self.states = json.load(f)
        return self.score

    def load_results(self) -> float:

        self.best_genome = self.trainer.load_genome(self.genome_path)
        with open(self.states_path) as f:
            self.states = json.load(f)

        if self.states:
            apples = self.states[-1].get('score', 0)
            steps  = len(self.states)
        else:
            apples, steps = 0, 0
        self.score = self.evaluator.evaluate(apples, steps)
        return self.score

    def replay(self, delay: float = 0.1) -> None:
        self.game_play.replay(self.states_path, delay)


class BalancedEvaluator:
    def __init__(self, apple_weight: float = 100.0, time_weight: float = 1.0):
        self.apple_weight = apple_weight
        self.time_weight = time_weight

    def evaluate(self, apples: int, time_steps: int) -> float:
        return apples * self.apple_weight + time_steps * self.time_weight

class TimeDecayEvaluator:
    def __init__(self, apple_weight: float = 100.0, time_weight: float = 10.0):
        self.apple_weight = apple_weight
        self.time_weight = time_weight

    def evaluate(self, apples: int, time_steps: int) -> float:
        import math
        return apples * self.apple_weight + math.sqrt(time_steps) * self.time_weight

class ThresholdTimeEvaluator:
    def __init__(self, apple_weight: float = 100.0, time_weight: float = 1.0):
        self.apple_weight = apple_weight
        self.time_weight = time_weight

    def evaluate(self, apples: int, time_steps: int) -> float:
        base = apples * self.apple_weight
        if apples > 0:
            return base + time_steps * self.time_weight
        return base

class ApplePriorityEvaluator:
    def __init__(self, apple_weight: float = 200.0, time_weight: float = 0.1):
        self.apple_weight = apple_weight
        self.time_weight = time_weight

    def evaluate(self, apples: int, time_steps: int) -> float:
        return apples * self.apple_weight + time_steps * self.time_weight
    
class OnlyApple:
    def __init__(self, apple_weight: float = 1, time_weight: float = 0.0):
        self.apple_weight = apple_weight
        self.time_weight = time_weight

    def evaluate(self, apples: int, time_steps: int) -> float:
        return apples * self.apple_weight + time_steps * self.time_weight

EVALUATORS = {
    'balanced': BalancedEvaluator,
    'time_decay': TimeDecayEvaluator,
    'threshold': ThresholdTimeEvaluator,
    'apple_priority': ApplePriorityEvaluator,
    'only_apple': OnlyApple
}