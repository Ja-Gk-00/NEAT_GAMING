import json
from typing import Optional

class InitialArchitecture:
    def __init__(self, input_size: int, output_size: int, hidden_layers: Optional[list] = None):
        self.input_size = input_size
        self.hidden_layers = hidden_layers or []
        self.output_size = output_size

    @classmethod
    def from_file(cls, path: str) -> 'InitialArchitecture':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            input_size=data['input_size'],
            output_size=data['output_size'],
            hidden_layers=data.get('hidden_layers', [])
        )

    def apply_to_config(self, config):
        genome_cfg = config.genome_config
        # Set sizes
        genome_cfg.num_inputs = self.input_size
        genome_cfg.num_outputs = self.output_size
        genome_cfg.num_hidden = sum(self.hidden_layers)
        # Ensure feed-forward topology
        genome_cfg.feed_forward = True
        genome_cfg.initial_connection = 'full'
        # Update input and output keys
        genome_cfg.input_keys = [-(i+1) for i in range(self.input_size)]
        genome_cfg.output_keys = list(range(self.output_size))
