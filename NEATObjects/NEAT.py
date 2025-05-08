import pickle
import neat
import json
from neat.reporting import StdOutReporter
from neat.statistics import StatisticsReporter
from neat.checkpoint import Checkpointer
from neat.nn import FeedForwardNetwork
from GameObjects.Snake import SnakeGame

class NEATTrainer:
    def __init__(
        self,
        config_path: str,
        game_train: SnakeGame,
        game_play: SnakeGame,
        evaluator
    ):
        # Load NEAT config
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        # Attach reporter and checkpoint
        self.pop = neat.Population(self.config)
        self.pop.add_reporter(StdOutReporter(True))
        self.pop.add_reporter(StatisticsReporter())
        self.pop.add_reporter(Checkpointer(
            generation_interval=10,
            filename_prefix='neat-checkpoint-'
        ))

        self.game_train = game_train
        self.game_play  = game_play
        self.evaluator  = evaluator

    def eval_genomes(self, genomes, config) -> None:
        for _, genome in genomes:
            net = FeedForwardNetwork.create(genome, config)
            # Run headless
            self.game_train.reset()
            steps = 0
            while steps < 1000 and not self.game_train.done:
                inputs = self.game_train.get_state()
                outputs = net.activate(inputs)
                action = int(outputs.index(max(outputs)))
                self.game_train.step(action, render=False)
                steps += 1
            apples = self.game_train.score
            fitness = self.evaluator.evaluate(apples, steps)
            genome.fitness = fitness

    def learn(self, generations: int):
        winner = self.pop.run(self.eval_genomes, generations)
        return winner

    def play(self, genome, max_steps: int = 1000, render: bool = True, states_path: str = None):
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
