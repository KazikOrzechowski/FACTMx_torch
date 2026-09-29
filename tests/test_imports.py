from factmx_torch import FACTMxModel, MissingViewFACTMx, __version__
from methods.factmx_torch import FACTMxModel as LegacyFACTMxModel


def test_public_and_legacy_imports() -> None:
    assert __version__ == "0.1.0"
    assert LegacyFACTMxModel is FACTMxModel
    assert issubclass(MissingViewFACTMx, FACTMxModel)
