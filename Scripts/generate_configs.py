import os
import configparser
import argparse
from typing import List, Tuple

# original defaults
DEFAULT_GRID_SIZES: List[Tuple[int,int]] = [(6,6), (10,10)]
DEFAULT_WEIGHT_MUT_RATES = [0.3, 0.5, 0.9]
DEFAULT_NODE_MUT_RATES = [0.05, 0.1, 0.25]
DEFAULT_SURVIVAL_THRESHOLDS = [0.85, 0.65, 0.25]
DEFAULT_COMPAT_COEFFS = [
    (1.0, 1.0),
    (0.25, 1.0),
    (1.0, 0.25),
]
DEFAULT_EVALUATORS = ['only_apple']
DEFAULT_BASE_CONFIG = 'initial.ini'

def parse_grid_sizes(ss: List[str]) -> List[Tuple[int,int]]:
    out = []
    for s in ss:
        w,h = s.split('x')
        out.append((int(w),int(h)))
    return out

def parse_compat_coeffs(ss: List[str]) -> List[Tuple[float,float]]:
    out = []
    for s in ss:
        a,b = s.split(',')
        out.append((float(a), float(b)))
    return out

def load_ref_parameters(path: str):
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        raise FileNotFoundError(f"Could not read reference config: {path}")
    # GAME
    gw = int(cfg['GAME']['grid_width'])
    gh = int(cfg['GAME']['grid_height'])
    # GENOME
    wmr = float(cfg['DefaultGenome']['weight_mutate_rate'])
    # assume weight_mutate_power left at default
    nadd = float(cfg['DefaultGenome']['node_add_prob'])
    ndele = float(cfg['DefaultGenome']['node_delete_prob'])
    # use average if they differ
    nmr = (nadd + ndele) / 2
    cd = float(cfg['DefaultGenome']['compatibility_disjoint_coefficient'])
    cw = float(cfg['DefaultGenome']['compatibility_weight_coefficient'])
    # REPRODUCTION
    surv = float(cfg['DefaultReproduction']['survival_threshold'])
    # EVALUATOR
    ev = cfg['EVALUATOR']['name']
    # ARCHITECTURE (optional)
    arch = cfg['ARCHITECTURE'].get('initial_architecture', '')
    return {
        'grid_sizes': [(gw,gh)],
        'weight_mut_rates': [wmr],
        'node_mut_rates': [nmr],
        'survival_thresholds': [surv],
        'compat_coeffs': [(cd,cw)],
        'evaluators': [ev],
        'arch_path': arch
    }

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
    evaluator_name: str,
    arch_path: str = ''
) -> None:
    parser = configparser.ConfigParser()
    read_files = parser.read(base_path)
    if not read_files:
        raise FileNotFoundError(f"Base config not found: {base_path}")
    for sec in ['DefaultGenome', 'DefaultReproduction']:
        if sec not in parser:
            raise KeyError(f"Missing section {sec} in base config")
    parser['DefaultGenome']['weight_mutate_rate'] = str(weight_rate)
    parser['DefaultGenome']['weight_mutate_power'] = '0.5'
    parser['DefaultGenome']['node_add_prob'] = str(node_rate)
    parser['DefaultGenome']['node_delete_prob'] = str(node_rate)
    parser['DefaultReproduction']['survival_threshold'] = str(survival)
    parser['DefaultGenome']['compatibility_disjoint_coefficient'] = str(compat_disjoint)
    parser['DefaultGenome']['compatibility_weight_coefficient'] = str(compat_weight)
    parser['GAME'] = {
        'grid_width': str(grid_width),
        'grid_height': str(grid_height),
        'cell_size': '20',
        'game_mode': '1'
    }
    parser['EVALUATOR'] = {'name': evaluator_name}
    parser['ARCHITECTURE'] = {'initial_architecture': arch_path}
    with open(out_path, 'w') as cfgfile:
        parser.write(cfgfile)

def main():
    p = argparse.ArgumentParser(
        description='Generate NEAT+GAME config files, optionally using a reference config for defaults.'
    )
    p.add_argument('-o','--output-dir', default='configs',
                   help='Directory to save generated config files.')
    p.add_argument('-b','--base-config', default=DEFAULT_BASE_CONFIG,
                   help='Path to base NEAT config file.')
    p.add_argument('-r','--ref-config',
                   help='Path to an existing config to pull defaults from.')
    p.add_argument('--grid-sizes', nargs='+',
                   help='Grid sizes (e.g. 6x6 10x10).')
    p.add_argument('--weight-mut-rates', nargs='+', type=float,
                   help='Weight mutation rates (e.g. 0.3 0.5).')
    p.add_argument('--node-mut-rates', nargs='+', type=float,
                   help='Node mutation rates (e.g. 0.05 0.1).')
    p.add_argument('--survival-thresholds', nargs='+', type=float,
                   help='Survival thresholds (e.g. 0.85 0.65).')
    p.add_argument('--compat-coeffs', nargs='+',
                   help='Compatibility coeffs as c_disjoint,c_weight (e.g. 1.0,1.0 0.25,1.0).')
    p.add_argument('--evaluators', nargs='+',
                   help='Evaluator names (e.g. apple_priority snake_score).')
    p.add_argument('--arch-dir', default='',
                   help='Directory containing initial architecture JSONs (optional).')
    args = p.parse_args()

    # load reference defaults if given
    ref = {}
    if args.ref_config:
        ref = load_ref_parameters(args.ref_config)

    # determine final parameter lists
    grid_sizes = (parse_grid_sizes(args.grid_sizes)
                  if args.grid_sizes
                  else ref.get('grid_sizes', DEFAULT_GRID_SIZES))
    weight_mut_rates = (args.weight_mut_rates
                        if args.weight_mut_rates is not None
                        else ref.get('weight_mut_rates', DEFAULT_WEIGHT_MUT_RATES))
    node_mut_rates = (args.node_mut_rates
                      if args.node_mut_rates is not None
                      else ref.get('node_mut_rates', DEFAULT_NODE_MUT_RATES))
    survival_thresholds = (args.survival_thresholds
                           if args.survival_thresholds is not None
                           else ref.get('survival_thresholds', DEFAULT_SURVIVAL_THRESHOLDS))
    compat_coeffs = (parse_compat_coeffs(args.compat_coeffs)
                     if args.compat_coeffs
                     else ref.get('compat_coeffs', DEFAULT_COMPAT_COEFFS))
    evaluators = (args.evaluators
                  if args.evaluators
                  else ref.get('evaluators', DEFAULT_EVALUATORS))
    arch_dir = args.arch_dir
    ref_arch = ref.get('arch_path', '')

    os.makedirs(args.output_dir, exist_ok=True)
    for gw, gh in grid_sizes:
        for wmr in weight_mut_rates:
            for nmr in node_mut_rates:
                for surv in survival_thresholds:
                    for cd, cw in compat_coeffs:
                        for eval_name in evaluators:
                            # pick architecture: prefer one in arch-dir, else ref, else empty
                            arch_name = ''
                            if arch_dir:
                                cand = os.path.join(arch_dir, f'arch_{gw}x{gh}.json')
                                if os.path.isfile(cand):
                                    arch_name = cand
                            elif ref_arch:
                                arch_name = ref_arch
                            exp = f'{gw}x{gh}_w{wmr}_n{nmr}_s{surv}_c{cd}-{cw}_{eval_name}'
                            out_path = os.path.join(args.output_dir, f'config_{exp}.ini')
                            make_config(
                                args.base_config,
                                out_path,
                                gw, gh, wmr, nmr, surv, cd, cw,
                                eval_name,
                                arch_name
                            )
                            print(f'Generated: {out_path}')

if __name__ == '__main__':
    main()
