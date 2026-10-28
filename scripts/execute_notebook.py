"""Execute from the project interpreter, using only the local kernel."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

root = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(root / 'work/matplotlib'))
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / 'work/jupyter-runtime')
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True, exist_ok=True)
path = root / 'outputs/garch_var_research.ipynb'
nb = nbformat.read(path, as_version=4)
manager = KernelSpecManager(kernel_dirs=[str(root / '.venv/share/jupyter/kernels')])
client = NotebookClient(nb, timeout=1200, kernel_name='garch-var', resources={'metadata': {'path': str(root)}},
                        kernel_manager_class='jupyter_client.manager.KernelManager')
# Set the spec manager explicitly so a global Anaconda kernel cannot be selected.
client.create_kernel_manager()
client.km.kernel_spec_manager = manager
client.execute()
nbformat.write(nb, path)
assert not any(out.output_type == 'error' for cell in nb.cells if cell.cell_type == 'code' for out in cell.outputs)
print('Notebook executed:', path)
