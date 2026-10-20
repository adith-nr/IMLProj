"""Optional baseline CLI; data preparation and later experiments live in notebooks."""
import argparse

from har_utils import (load_windows, feature_columns, models, evaluate,
                       record_environment, save_summary)


def main():
    parser = argparse.ArgumentParser(
        description='Optional baseline training after notebooks 01 and 02.')
    parser.add_argument('stage', nargs='?', default='baselines',
                        choices=['baselines'])
    parser.parse_args()
    X, meta = load_windows(2)
    X = X[feature_columns(X)]
    record_environment()
    for name, model in models().items():
        evaluate(X, meta, model, name, 'loso')
        save_summary()
    evaluate(X, meta, models()['RF'], 'RF', 'random')
    save_summary()


if __name__ == '__main__':
    main()
