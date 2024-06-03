import pkgutil
import importlib
import logging


net_filenames = [
    name for finder, name, _ in pkgutil.iter_modules(['net'])
]

_net_modules = [
    importlib.import_module('net.{}'.format(name)) for name in net_filenames
]


def create_net(conf):
    """Create model.

    Args:
        conf (dict): Configuration. It constains:
        model_type (str): Model type.
    """

    net_type = conf['net']['type']
    device = conf['device']

    # dynamic instantiation
    for module in _net_modules:
        net_cls = getattr(module, net_type, None)
        if net_cls is not None:
            break
    if net_cls is None:
        raise ValueError(f'Net {net_type} is not found.')
    logging.info(f'Net [{net_cls.__name__}] is imported.')

    return net_cls(conf).to(device)
