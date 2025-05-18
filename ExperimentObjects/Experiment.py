import graphlib
from typing import List, Tuple, Optional
from NEATObjects.EvalFunc import FitnessEvaluator
from NEATObjects.NEAT import NEATTrainer
from GameObjects.Snake import SnakeGame
from neat.checkpoint import Checkpointer

import neat
import json
import configparser
import os


from InitialArchitectureObjects.InitalArchitecture import InitialArchitecture

class Experiment:
    def __init__(
        self,
        config_path: str,
        output_dir: str = '.',
        generations: int = 225
    ):
        self.config_path = config_path
        self.generations = generations

        # Experiment directory
        cfg_name = os.path.splitext(os.path.basename(config_path))[0]
        self.exp_dir = os.path.join(output_dir, cfg_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Paths
        self.genome_path = os.path.join(self.exp_dir, 'best_genome.pkl')
        self.states_path = os.path.join(self.exp_dir, 'game_states.json')

        # Load config
        parser = configparser.ConfigParser()
        files = parser.read(config_path)
        if not files:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # GAME section
        if 'GAME' not in parser:
            raise KeyError("Missing [GAME] section in config file.")
        game_cfg = parser['GAME']
        gw = game_cfg.getint('grid_width', fallback=30)
        gh = game_cfg.getint('grid_height', fallback=30)
        cs = game_cfg.getint('cell_size', fallback=20)
        gm = game_cfg.getint('game_mode', fallback=1)

        # Instantiate games
        self.game_train = SnakeGame(gw, gh, cs, gm)
        self.game_play  = SnakeGame(gw, gh, cs, gm)

        # ARCHITECTURE section
        arch_path = parser.get('ARCHITECTURE', 'initial_architecture', fallback='').strip()
        if arch_path:
            self.initial_arch = InitialArchitecture.from_file(arch_path)
        else:
            # default: one input per cell, 4 outputs, no hidden
            input_size = gw * gh
            self.initial_arch = InitialArchitecture(input_size=input_size, output_size=4)

        # EVALUATOR section
        if 'EVALUATOR' not in parser:
            raise KeyError("Missing [EVALUATOR] section in config file.")
        eval_name = parser.get('EVALUATOR', 'name', fallback='balanced')
        if eval_name not in EVALUATORS:
            raise KeyError(f"Unknown evaluator: {eval_name}")
        self.evaluator = EVALUATORS[eval_name]()

        # Setup NEAT trainer
        self.trainer = NEATTrainer(
            config_path=self.config_path,
            game_train=self.game_train,
            game_play=self.game_play,
            evaluator=self.evaluator,
            initial_arch=self.initial_arch
        )

        # Redirect NEAT checkpoints
        reporters = self.trainer.pop.reporters.reporters
        # remove old Checkpointers
        old = [r for r in reporters if isinstance(r, Checkpointer)]
        for r in old:
            reporters.remove(r)
        # add new with exp_dir prefix
        prefix = os.path.join(self.exp_dir, 'neat-checkpoint-')
        self.trainer.pop.add_reporter(Checkpointer(generation_interval=10, filename_prefix=prefix))

        # Results
        self.best_genome: Optional[object] = None
        self.score: Optional[float] = None
        self.states: Optional[list] = None

    def run(self) -> float:
        self.best_genome = self.trainer.learn(self.generations)
        self.trainer.save_genome(self.best_genome, self.genome_path)
        self.score = self.trainer.play(
            self.best_genome,
            render=False,
            states_path=self.states_path
        )
        with open(self.states_path, 'r') as f:
            self.states = json.load(f)
        return self.score

    def load_results(self) -> float:
        self.best_genome = self.trainer.load_genome(self.genome_path)
        with open(self.states_path, 'r') as f:
            self.states = json.load(f)
        if self.states:
            apples = self.states[-1].get('score', 0)
            steps  = len(self.states)
        else:
            apples, steps = 0, 0
        self.score = self.evaluator.evaluate(apples, steps)
        return self.score

    def replay(self, delay: float = 0.2) -> None:
        self.game_play.replay(self.states_path, delay)

    def visualize_architecture(self,
                               filename: str = 'architecture',
                               view: bool = True) -> str:
        
        if not hasattr(self, 'best_genome') or self.best_genome is None:
            # attempt to load if we've already run
            self.best_genome = self.trainer.load_genome(self.genome_path)

        genome = self.best_genome
        cfg    = self.trainer.config
        dot = graphlib.Digraph(format='png')
        dot.attr('graph', rankdir='LR')

        inputs  = cfg.genome_config.input_keys
        outputs = cfg.genome_config.output_keys
        for nid in inputs:
            dot.node(str(nid), shape='circle', style='filled', fillcolor='lightblue', label=f'I{abs(nid)}')
        for nid in outputs:
            dot.node(str(nid), shape='doublecircle', style='filled', fillcolor='lightgreen', label=f'O{nid}')
        for nid, node in genome.nodes.items():
            if nid not in inputs and nid not in outputs:
                dot.node(str(nid), shape='circle', label=str(nid))

        for cg in genome.connections.values():
            if not cg.enabled:
                continue
            src, dst = cg.key
            weight = cg.weight
            dot.edge(str(src), str(dst), label=f'{weight:.2f}', penwidth=str(max(0.1, abs(weight) * 2)))

        outpath = os.path.join(self.exp_dir, filename)
        dot.render(outpath, view=view)
        # graphviz appends “.png”
        return outpath + '.png'


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