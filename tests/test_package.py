import subprocess
import sys

import pytest


def test_public_api():
    import pool_segmentation
    from pool_segmentation import PoolModel, find_model, water_polygons

    assert PoolModel.__module__ == "pool_segmentation.model.predict"
    assert callable(find_model) and callable(water_polygons)
    with pytest.raises(AttributeError):
        pool_segmentation.Unknown


@pytest.mark.parametrize("module", ["pool_segmentation", "pool_segmentation.labeling.commands", "pool_segmentation.app.server"])
def test_tools_that_need_no_model_do_not_import_torch(module):
    # torch takes seconds to import: the labeling commands and the Labels page must start without it (and without
    # huggingface_hub: only when the published model is downloaded)
    code = f"import sys, {module}; print('torch' in sys.modules, 'huggingface_hub' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False False"
