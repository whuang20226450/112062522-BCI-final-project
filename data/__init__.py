import pkgutil
import importlib
import logging


model_filenames = [
    name for finder, name, _ in pkgutil.iter_modules(['data'])
]

_model_modules = [
    importlib.import_module('data.{}'.format(name)) for name in model_filenames
]


def create_loaderHandler(conf, mode):
    """Create model.

    Args:
        conf (dict): Configuration. It constains:
        model_type (str): Model type.
    """

    dataloader_type = conf['dataloader']['type']

    # dynamic instantiation
    for module in _model_modules:
        dataloader_cls = getattr(module, dataloader_type, None)
        if dataloader_cls is not None:
            break
    if dataloader_cls is None:
        raise ValueError(f'Model {dataloader_type} is not found.')

    dataloader_handler = dataloader_cls(conf, mode)

    logging.info(f'dataloader [{dataloader_cls.__name__}] is created.')

    return dataloader_handler
