

def get_trainable_class(*args, **kwargs):
    from .main import ExperimentRunner
    return ExperimentRunner


def get_variant_spec(command_line_args, *args, **kwargs):
    from .variants import get_variant_spec
    variant_spec = get_variant_spec(command_line_args, *args, **kwargs)
    return variant_spec


def get_parser():
    from examples.utils import get_parser
    parser = get_parser()
    return parser
