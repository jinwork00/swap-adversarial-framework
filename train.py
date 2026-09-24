import argparse
from dataclasses import fields
import json
from pathlib import Path
from src.training import TrainingConfig, fit


def make_parser():
    parser = argparse.ArgumentParser(description='Single-run CLI using the common augmentation-enabled research defaults.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-config', type=Path, required=True, help='JSON data_list/domain_list configuration')
    parser.add_argument('--data-root', type=Path, default=Path('.'), help='Base path for signal files')
    parser.add_argument('--output-dir', type=Path, help='New output folder; defaults to a timestamped folder in runs/')
    defaults = TrainingConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        option = '--' + field.name.replace('_', '-')
        kwargs = {'default': value, 'help': field.name.replace('_', ' ')}
        if isinstance(value, bool):
            kwargs['action'] = argparse.BooleanOptionalAction
        elif isinstance(value, tuple):
            kwargs.update(type=float, nargs='+')
        else:
            kwargs['type'] = type(value)
        parser.add_argument(option, **kwargs)
    return parser


def main():
    args = make_parser().parse_args()
    values = {field.name: getattr(args, field.name) for field in fields(TrainingConfig)}
    values['class_weights'] = tuple(values['class_weights'])
    config = TrainingConfig(**values)
    with args.data_config.open() as stream:
        data_config = json.load(stream)
    output = fit(data_config, args.data_root, args.output_dir, config)
    print(f'Outputs: {output}')


if __name__ == '__main__':
    main()
