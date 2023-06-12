from muse.datasets.preprocess.data_preprocessor import DataPreprocessor
from muse.models.bc.hydra.helpers import get_mode_smooth_preproc_fn
from attrdict import AttrDict as d


export = d(
    cls=DataPreprocessor,
    name="cs_smooth_preprocessor",
    episode_preproc_fn=get_mode_smooth_preproc_fn("click_state", "smooth_click_state"),
)
