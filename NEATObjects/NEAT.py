import pickle
import neat
import json
import math
from typing import Optional, List

from neat.reporting import StdOutReporter
from neat.statistics import StatisticsReporter
from neat.checkpoint import Checkpointer
from neat.nn import FeedForwardNetwork
from GameObjects.Snake import SnakeGame

from GameObjects.Snake import UP, DOWN, LEFT, RIGHT

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

        #num_inputs = len(game_train.get_state())
        #self.config.genome_config.num_inputs  = num_inputs

        # Instantiate evaluator
        self.evaluator = evaluator

    def eval_genomes(self, genomes, config) -> None:
        dirs = [UP, DOWN, LEFT, RIGHT]

        for _, genome in genomes:
            net = FeedForwardNetwork.create(genome, config)
            genome.fitness = 0.0
            self.game_train.reset()

            prev_dist = self._dist_to_apple(self.game_train)
            max_steps = 1000

            for step in range(max_steps):
                if self.game_train.done:
                    genome.fitness -= 1.0  # death penalty
                    break

                # Base sensor state
                state = self.game_train.get_state()  # sensor vector

                # Append normalized delta-to-apple
                hx, hy = self.game_train.snake.head
                ax, ay = self.game_train.apples[0]
                dx = (ax - hx) / (self.game_train.grid_width  - 1)
                dy = (ay - hy) / (self.game_train.grid_height - 1)
                inputs = state + [dx, dy]

                outputs = net.activate(inputs)

                # Prevent immediate reverse
                curr_i = dirs.index(self.game_train.snake.direction)
                outputs[curr_i ^ 1] = -float('inf')

                action = int(outputs.index(max(outputs)))
                self.game_train.step(action)

                # Shaped reward: proximity bonus
                curr_dist = self._dist_to_apple(self.game_train)
                genome.fitness += (prev_dist - curr_dist) * 0.1
                prev_dist = curr_dist

                # Bonus on apple
                if self.game_train.score > 0:
                    genome.fitness += 5.0

            # Time penalty
            genome.fitness -= 0.001 * step

    def learn(self, generations: int):
        return self.pop.run(self.eval_genomes, generations)

    def play(self, genome, max_steps: int = 1000, render: bool = True, states_path: str = None):
        # Create network
        net = FeedForwardNetwork.create(genome, self.config)
        self.game_play.reset()

        dirs = [UP, DOWN, LEFT, RIGHT]
        steps = 0
        states = []

        while steps < max_steps and not self.game_play.done:
            # Base sensor inputs
            state = self.game_play.get_state()  # length N

            # Append normalized vector-to-apple (2 values)
            hx, hy = self.game_play.snake.head
            ax, ay = self.game_play.apples[0]
            dx = (ax - hx) / (self.game_play.grid_width  - 1)
            dy = (ay - hy) / (self.game_play.grid_height - 1)
            inputs = state + [dx, dy]            # length N+2 == config.num_inputs

            # Activate and mask reverse
            outputs = net.activate(inputs)
            curr_i = dirs.index(self.game_play.snake.direction)
            outputs[curr_i ^ 1] = -float('inf')

            # Select action and step
            action = int(outputs.index(max(outputs)))
            self.game_play.step(action, render=render)

            # Record state for replay
            states.append({
                'snake':  list(self.game_play.snake.body),
                'apples': list(self.game_play.apples),
                'score':  self.game_play.score
            })
            steps += 1

        # Optionally save states
        if states_path:
            with open(states_path, 'w') as f:
                json.dump(states, f, indent=2)

        # Return final performance
        return self.evaluator.evaluate(self.game_play.score, steps)

    def _dist_to_apple(self, game: SnakeGame) -> float:
        hx, hy = game.snake.head
        ax, ay = game.apples[0]
        return math.hypot(hx - ax, hy - ay)
    
    def return_genomes(self) -> List[neat.DefaultGenome]:
        genomes = list(self.pop.population)
        genomes.sort(key=lambda g: getattr(g, 'fitness', int('-1_000_000')), reverse=True)
        return genomes


    def save_genome(self, genome, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(genome, f)

    def load_genome(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
        