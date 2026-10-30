from sglang._build_version import (
    BASE_VERSION,
    SCM_VERSION,
    version_tuple,
    with_base_version,
)

try:
    from sglang._version import __version__
except ImportError:
    try:
        import importlib.metadata

        __version__ = importlib.metadata.version("sglang")
    except Exception:
        try:
            import pathlib

            from setuptools_scm import get_version

            # point to the directory containing pyproject.toml.
            project_root = pathlib.Path(__file__).parent.parent.parent
            __version__ = get_version(
                root=str(project_root),
                **SCM_VERSION,
            )
        except Exception:
            # Fallback for development without build
            __version__ = BASE_VERSION

__version__ = with_base_version(__version__)
__version_tuple__ = version_tuple(__version__)
