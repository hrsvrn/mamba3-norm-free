__version__ = "2.3.2.post1"

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn = None
    mamba_inner_fn = None

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None

try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    Mamba2 = None

from mamba_ssm.modules.mamba3 import Mamba3

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    MambaLMHeadModel = None
