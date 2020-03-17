import yaml

from graph_ter_cls.runner import BackboneRunner
from graph_ter_cls.runner import ClassifierRunner
from graph_ter_cls.runner import EvaluationRunner
# from graph_ter_cls.runner import SVMClassifierRunner
from graph_ter_cls.tools.configuration import get_parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    args = parser.parse_args()

    if args.phase == 'backbone':
        runner = BackboneRunner(args)
    elif args.phase == 'classifier':
        runner = ClassifierRunner(args)
    # elif args.phase == 'svm':
    #     runner = SVMClassifierRunner(args)
    elif args.phase == 'test':
        runner = EvaluationRunner(args)
    else:
        raise ValueError('Unknown phase.')

    runner.run()


if __name__ == '__main__':
    main()
