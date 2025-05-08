import pickle
import neat
import json
from typing import Optional

from neat.reporting import StdOutReporter
from neat.statistics import StatisticsReporter
from neat.checkpoint import Checkpointer
from neat.nn import FeedForwardNetwork
from GameObjects.Snake import SnakeGame

from InitialArchitectureObjects.InitalArchitecture import InitialArchitecture

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
        return base + time_steps * self.time_weight if apples > 0 else base

class ApplePriorityEvaluator:
    def __init__(self, apple_weight: float = 200.0, time_weight: float = 0.1):
        self.apple_weight = apple_weight
        self.time_weight = time_weight
    def evaluate(self, apples: int, time_steps: int) -> float:
        return apples * self.apple_weight + time_steps * self.time_weight

# Mapping evaluator names to classes
EVALUATORS = {
    'balanced': BalancedEvaluator,
    'time_decay': TimeDecayEvaluator,
    'threshold': ThresholdTimeEvaluator,
    'apple_priority': ApplePriorityEvaluator
}

class NEATTrainer:
    def __init__(
        self,
        config_path: str,
        game_train: SnakeGame,
        game_play: SnakeGame,
        evaluator,
        initial_arch: Optional[InitialArchitecture] = None
    ):
        # Load NEAT config
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        # Apply initial architecture if provided
        if initial_arch:
            initial_arch.apply_to_config(self.config)

        # Create population with reporters
        self.pop = neat.Population(self.config)
        self.pop.add_reporter(StdOutReporter(True))
        self.pop.add_reporter(StatisticsReporter())
        self.pop.add_reporter(Checkpointer(
            generation_interval=10,
            filename_prefix='neat-checkpoint-'
        ))

        self.game_train = game_train
        self.game_play  = game_play
        # Instantiate evaluator
        self.evaluator = evaluator

    def eval_genomes(self, genomes, config) -> None:
        for _, genome in genomes:
            net = FeedForwardNetwork.create(genome, config)
            self.game_train.reset()
            steps = 0
            while steps < 1000 and not self.game_train.done:
                inputs = self.game_train.get_state()
                outputs = net.activate(inputs)
                action = int(outputs.index(max(outputs)))
                self.game_train.step(action, render=False)
                steps += 1
            apples = self.game_train.score
            genome.fitness = self.evaluator.evaluate(apples, steps)

    def learn(self, generations: int):
        return self.pop.run(self.eval_genomes, generations)

    def play(self, genome, max_steps: int = 500, render: bool = True, states_path: str = None):
        net = FeedForwardNetwork.create(genome, self.config)
        self.game_play.reset()
        steps = 0
        states = []
        while steps < max_steps and not self.game_play.done:
            inputs = self.game_play.get_state()
            outputs = net.activate(inputs)
            action = int(outputs.index(max(outputs)))
            self.game_play.step(action, render=render)
            states.append({'snake': list(self.game_play.snake.body),
                           'apples': list(self.game_play.apples),
                           'score': self.game_play.score})
            steps += 1
        if states_path:
            with open(states_path, 'w') as f:
                json.dump(states, f, indent=2)
        return self.evaluator.evaluate(self.game_play.score, steps)

    def save_genome(self, genome, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(genome, f)

    def load_genome(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)