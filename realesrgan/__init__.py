"""Top-level package exports for Real-ESRGAN utilities.

在源码仓库直接运行时，`version.py` 可能不存在（通常由打包流程生成）。
这里采用容错导出，保证 `from realesrgan import RealESRGANer` 可用。
"""

# flake8: noqa

from .utils import RealESRGANer

try:
	from .version import __version__
except Exception:
	__version__ = "0.0.0"

__all__ = ["RealESRGANer", "__version__"]
