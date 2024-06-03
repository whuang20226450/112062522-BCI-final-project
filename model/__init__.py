import pkgutil
import importlib
import logging


model_filenames = [
    name for finder, name, _ in pkgutil.iter_modules(['model'])
]

_model_modules = [
    importlib.import_module('model.{}'.format(name)) for name in model_filenames
]


def create_model(conf):
    """Create model.

    Args:
        conf (dict): Configuration. It constains:
        model_type (str): Model type.
    """

    model_type = conf['model']

    # dynamic instantiation
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(conf)

    logging.info(f'Model [{model.__class__.__name__}] is created.')

    return model
