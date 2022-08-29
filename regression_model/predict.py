import typing as t

import pandas as pd

from regression_model import __version__ as _version
from regression_model.config.core import config
from regression_model.processing.data_manager import load_pipeline
#from regression_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    #data = load_dataset(file_name=config.app_config.test_data_file)
    #validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version}
    
    predictions = _pipe.predict(
            X=data[config.model_config.features]
        )
    
    results = {
        "predictions": [pred for pred in predictions],  # type: ignore
        "version": _version
        }
       
    return results
