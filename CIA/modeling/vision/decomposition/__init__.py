from .bayes_decomposition import *
from .build import *


def build_decomposition(config, **kwargs):
    model_name = config['MODEL']['DECOMPOSITION']['NAME']
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, **kwargs)