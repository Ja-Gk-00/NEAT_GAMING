import os
import configparser
import argparse
from typing import List, Tuple

grid_sizes: List[Tuple[int,int]] = [(5,5), (10,10), (30,30)]
weight_mut_rates = [0.3, 0.5, 0.9]
node_mut_rates = [0.05, 0.1, 0.25]
survival_thresholds = [0.85, 0.65, 0.25]
compat_coeffs = [
    (1.0, 1.0),
    (0.25, 1.0),
    (1.0, 0.25),
]

evaluators = ['balanced', 'time_decay', 'threshold', 'apple_priority']

DEFAULT_BASE_CONFIG = 'initial.ini'

def make_config(
    base_path: str,
    out_path: str,
    grid_width: int,
    grid_height: int,
    weight_rate: float,
    node_rate: float,
    survival: float,
    compat_disjoint: float,
    compat_weight: float,
    evaluator_name: str
) -> None:
    parser = configparser.ConfigParser()
    read = parser.read(base_path)
    if not read:
        raise FileNotFoundError(f"Base config not found: {base_path}")
    # Ensure sections exist
    for s in ['DefaultGenome', 'DefaultReproduction']:
        if s not in parser:
            raise KeyError(f"Missing section {s} in base config")
    # Override NEAT params
    parser['DefaultGenome']['weight_mutate_rate'] = str(weight_rate)
    parser['DefaultGenome']['weight_mutate_power'] = '0.5'
    parser['DefaultGenome']['node_add_prob'] = str(node_rate)
    parser['DefaultGenome']['node_delete_prob'] = str(node_rate)
    parser['DefaultReproduction']['survival_threshold'] = str(survival)
    parser['DefaultGenome']['compatibility_disjoint_coefficient'] = str(compat_disjoint)
    parser['DefaultGenome']['compatibility_weight_coefficient'] = str(compat_weight)
    # Add GAME section
    parser['GAME'] = {
        'grid_width': str(grid_width),
        'grid_height': str(grid_height),
        'cell_size': '20',
        'game_mode': '1'
    }
    # Add EVALUATOR section
    parser['EVALUATOR'] = {'name': evaluator_name}
    # Write
    with open(out_path, 'w') as cfgfile:
        parser.write(cfgfile)

def main():
    parser = argparse.ArgumentParser(
        description='Generate NEAT+GAME config files with evaluators.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='configs',
        help='Directory to save generated config files.'
    )
    parser.add_argument(
        '--base-config', '-b',
        default=DEFAULT_BASE_CONFIG,
        help='Path to base NEAT config file.'
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    base_cfg = args.base_config
    os.makedirs(out_dir, exist_ok=True)

    for gw, gh in grid_sizes:
        for wmr in weight_mut_rates:
            for nmr in node_mut_rates:
                for surv in survival_thresholds:
                    for cd, cw in compat_coeffs:
                        for eval_name in evaluators:
                            exp_name = f'{gw}x{gh}_w{wmr}_n{nmr}_s{surv}_c{cd}-{cw}_{eval_name}'
                            cfg_path = os.path.join(out_dir, f'config_{exp_name}.ini')
                            make_config(
                                base_cfg,
                                cfg_path,
                                gw, gh,
                                wmr, nmr, surv, cd, cw,
                                eval_name
                            )
                            print(f'Generated config: {cfg_path}')

if __name__ == '__main__':
    main()
