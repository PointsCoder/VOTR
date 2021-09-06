from .detector3d_template import Detector3DTemplate
from .votr_ssd_net import VoTrSSD
from .votr_tsd_net import VoTrRCNN

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'VoTrSSD': VoTrSSD,
    'VoTrRCNN': VoTrRCNN,
}

def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model