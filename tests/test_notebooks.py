from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

# Locate the notebooks directory
notebook_dir = Path(__file__).parent.parent / "src" / "notebooks"
notebook_files = list(notebook_dir.glob("*.ipynb"))


@pytest.mark.parametrize("notebook_path", notebook_files, ids=lambda p: p.name)
def test_notebook_execution(notebook_path):
    # Load the notebook file
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    project_root = Path(__file__).parent.parent

    try:
        ep.preprocess(nb, {"metadata": {"path": str(project_root)}})
    except CellExecutionError as e:
        pytest.fail(f"Notebook {notebook_path.name} failed cell execution:\n{e}")
