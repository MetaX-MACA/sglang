import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import keyword
import os
import stat
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

_REMOTE_VAE_PACKAGE_PREFIX = "_sglang_remote_vae"
_REMOTE_VAE_IMPORT_LOCK = threading.RLock()
_REMOTE_VAE_FINDERS: dict[str, "_RemoteVAESnapshotFinder"] = {}


@dataclass(frozen=True)
class _RemoteVAESourceSnapshot:
    """Immutable Python sources belonging to one remote VAE component."""

    component_root: Path
    sources: Mapping[str, bytes]
    directories: frozenset[str]
    package_name: str


def _read_remote_vae_source(path: Path, relative_path: str) -> bytes:
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"Cannot snapshot remote VAE Python source {relative_path!r}"
        ) from exc
    return source


def _snapshot_remote_vae_sources(component_root: Path) -> _RemoteVAESourceSnapshot:
    """Read a component's Python package once, without following directories."""
    try:
        canonical_root = component_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"Remote VAE component directory does not exist: {component_root}"
        ) from exc
    if not canonical_root.is_dir():
        raise ValueError(
            f"Remote VAE component path is not a directory: {canonical_root}"
        )

    sources: dict[str, bytes] = {}
    directories: set[str] = set()

    def scan_directory(directory: Path, relative_directory: PurePosixPath) -> None:
        try:
            entries = sorted(
                os.scandir(directory), key=lambda entry: os.fsencode(entry.name)
            )
        except OSError as exc:
            relative_name = relative_directory.as_posix()
            raise ValueError(
                f"Cannot scan remote VAE directory {relative_name!r}"
            ) from exc

        for entry in entries:
            relative_path = relative_directory / entry.name
            relative_name = relative_path.as_posix()
            lexical_path = directory / entry.name

            if entry.is_symlink():
                try:
                    target_stat = entry.stat(follow_symlinks=True)
                except OSError as exc:
                    if lexical_path.suffix == ".py":
                        raise ValueError(
                            "Remote VAE Python symlink is broken or cyclic: "
                            f"{relative_name}"
                        ) from exc
                    continue
                if stat.S_ISDIR(target_stat.st_mode):
                    raise ValueError(
                        f"Remote VAE directory symlinks are not allowed: {relative_name}"
                    )
                if lexical_path.suffix == ".py":
                    if not stat.S_ISREG(target_stat.st_mode):
                        raise ValueError(
                            "Remote VAE Python symlink must target a regular file: "
                            f"{relative_name}"
                        )
                    sources[relative_name] = _read_remote_vae_source(
                        lexical_path, relative_name
                    )
                continue

            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(
                    f"Cannot inspect remote VAE path {relative_name!r}"
                ) from exc
            if stat.S_ISDIR(entry_stat.st_mode):
                directories.add(relative_name)
                scan_directory(lexical_path, relative_path)
            elif lexical_path.suffix == ".py":
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ValueError(
                        "Remote VAE Python source must be a regular file: "
                        f"{relative_name}"
                    )
                sources[relative_name] = _read_remote_vae_source(
                    lexical_path, relative_name
                )

    scan_directory(canonical_root, PurePosixPath())

    digest = hashlib.sha256()

    def update_digest(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, byteorder="big"))
        digest.update(value)

    update_digest(b"root")
    update_digest(os.fsencode(canonical_root))
    for relative_name in sorted(directories, key=os.fsencode):
        update_digest(b"directory")
        update_digest(os.fsencode(relative_name))
    for relative_name in sorted(sources, key=os.fsencode):
        update_digest(b"source")
        update_digest(os.fsencode(relative_name))
        update_digest(sources[relative_name])
    package_name = f"{_REMOTE_VAE_PACKAGE_PREFIX}_{digest.hexdigest()}"
    return _RemoteVAESourceSnapshot(
        component_root=canonical_root,
        sources=MappingProxyType(sources),
        directories=frozenset(directories),
        package_name=package_name,
    )


class _RemoteVAESnapshotSourceLoader(importlib.abc.SourceLoader):
    """Compile a module strictly from bytes captured in a VAE snapshot."""

    def __init__(
        self,
        *,
        fullname: str,
        filename: str,
        source: bytes,
        is_package: bool,
    ) -> None:
        self._fullname = fullname
        self._filename = filename
        self._source = source
        self._is_package = is_package

    def get_filename(self, fullname: str) -> str:
        if fullname != self._fullname:
            raise ImportError(f"Snapshot loader cannot load {fullname!r}")
        return self._filename

    def get_data(self, path: str) -> bytes:
        if os.fspath(path) != self._filename:
            raise OSError(f"Path is outside the remote VAE snapshot: {path}")
        return self._source

    def get_code(self, fullname: str):
        filename = self.get_filename(fullname)
        return self.source_to_code(self._source, filename)

    def get_source(self, fullname: str) -> str:
        self.get_filename(fullname)
        return importlib.util.decode_source(self._source)

    def is_package(self, fullname: str) -> bool:
        self.get_filename(fullname)
        return self._is_package


class _RemoteVAESnapshotFinder(importlib.abc.MetaPathFinder):
    """Resolve only one synthetic package from its immutable source snapshot."""

    def __init__(self, snapshot: _RemoteVAESourceSnapshot) -> None:
        self.snapshot = snapshot
        self._package_path = f"<{snapshot.package_name}:snapshot>"

    def _source_spec(self, fullname: str, relative_name: str, *, is_package: bool):
        filename = str(self.snapshot.component_root / relative_name)
        loader = _RemoteVAESnapshotSourceLoader(
            fullname=fullname,
            filename=filename,
            source=self.snapshot.sources[relative_name],
            is_package=is_package,
        )
        spec = importlib.util.spec_from_loader(
            fullname, loader, origin=filename, is_package=is_package
        )
        if spec is not None and is_package:
            spec.submodule_search_locations = [self._package_path]
        return spec

    def _namespace_spec(self, fullname: str):
        spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
        spec.submodule_search_locations = [self._package_path]
        return spec

    def find_spec(self, fullname: str, path=None, target=None):
        package_name = self.snapshot.package_name
        if fullname == package_name:
            if "__init__.py" in self.snapshot.sources:
                return self._source_spec(fullname, "__init__.py", is_package=True)
            return self._namespace_spec(fullname)

        prefix = f"{package_name}."
        if not fullname.startswith(prefix):
            return None
        relative_module = fullname[len(prefix) :].replace(".", "/")
        package_source = f"{relative_module}/__init__.py"
        module_source = f"{relative_module}.py"
        if package_source in self.snapshot.sources:
            return self._source_spec(fullname, package_source, is_package=True)
        if module_source in self.snapshot.sources:
            return self._source_spec(fullname, module_source, is_package=False)
        if relative_module in self.snapshot.directories:
            return self._namespace_spec(fullname)
        return None

    def invalidate_caches(self) -> None:
        return None


def _remote_vae_package_name(component_root: Path) -> str:
    """Return a stable, content-addressed package name for remote VAE code."""
    return _snapshot_remote_vae_sources(component_root).package_name


def _uninstall_remote_vae_finder(package_name: str) -> None:
    finder = _REMOTE_VAE_FINDERS.pop(package_name, None)
    if finder is not None:
        sys.meta_path[:] = [
            candidate for candidate in sys.meta_path if candidate is not finder
        ]


def _remove_remote_vae_modules(module_names: set[str]) -> None:
    """Remove imported modules and the attributes importlib set on parents."""
    with _REMOTE_VAE_IMPORT_LOCK:
        package_names = {
            module_name
            for module_name in module_names
            if module_name.startswith(f"{_REMOTE_VAE_PACKAGE_PREFIX}_")
            and "." not in module_name
        }
        for module_name in sorted(
            module_names, key=lambda name: name.count("."), reverse=True
        ):
            module = sys.modules.get(module_name)
            parent_name, separator, child_name = module_name.rpartition(".")
            if separator:
                parent = sys.modules.get(parent_name)
                if parent is not None and getattr(parent, child_name, None) is module:
                    delattr(parent, child_name)
            sys.modules.pop(module_name, None)
        for package_name in package_names:
            _uninstall_remote_vae_finder(package_name)


def _remove_remote_vae_package(package_name: str) -> None:
    module_names = {
        name
        for name in sys.modules
        if name == package_name or name.startswith(f"{package_name}.")
    }
    module_names.add(package_name)
    _remove_remote_vae_modules(module_names)


def _validate_remote_vae_class_reference(class_reference: object) -> tuple[str, str]:
    if not isinstance(class_reference, str) or "." not in class_reference:
        raise ValueError(
            f"VAE auto_map entry must be '<module>.<class>', got {class_reference!r}"
        )
    module_path, class_name = class_reference.rsplit(".", 1)
    reference_parts = (*module_path.split("."), class_name)
    if not all(
        part.isidentifier() and not keyword.iskeyword(part) for part in reference_parts
    ):
        raise ValueError(f"Invalid VAE auto_map entry: {class_reference!r}")
    return module_path, class_name


def _get_remote_vae_auto_model_reference(config: Mapping) -> object | None:
    if "auto_map" not in config:
        return None
    auto_map = config["auto_map"]
    if not isinstance(auto_map, Mapping):
        raise ValueError("VAE auto_map must be a mapping")
    if "AutoModel" not in auto_map:
        return None
    class_reference = auto_map["AutoModel"]
    _validate_remote_vae_class_reference(class_reference)
    return class_reference


def _ensure_remote_vae_finder(
    snapshot: _RemoteVAESourceSnapshot,
) -> _RemoteVAESnapshotFinder:
    package_name = snapshot.package_name
    finder = _REMOTE_VAE_FINDERS.get(package_name)
    if finder is not None:
        if (
            finder.snapshot.component_root != snapshot.component_root
            or finder.snapshot.sources != snapshot.sources
            or finder.snapshot.directories != snapshot.directories
        ):
            raise RuntimeError(f"Remote VAE package hash collision: {package_name}")
        if finder not in sys.meta_path:
            sys.meta_path.insert(0, finder)
        return finder
    if package_name in sys.modules:
        raise RuntimeError(
            f"Remote VAE package name is already occupied: {package_name}"
        )
    finder = _RemoteVAESnapshotFinder(snapshot)
    _REMOTE_VAE_FINDERS[package_name] = finder
    sys.meta_path.insert(0, finder)
    return finder


def _load_remote_vae_class(
    component_model_path: str,
    class_reference: object,
    *,
    trust_remote_code: bool,
) -> type:
    """Load a remote VAE class without a filesystem-backed import path."""
    if not trust_remote_code:
        raise ValueError(
            "Loading a VAE class from auto_map requires trust_remote_code=True"
        )
    module_path, class_name = _validate_remote_vae_class_reference(class_reference)
    snapshot = _snapshot_remote_vae_sources(Path(component_model_path))
    package_name = snapshot.package_name
    qualified_module_name = f"{package_name}.{module_path}"

    with _REMOTE_VAE_IMPORT_LOCK:
        package_created = package_name not in sys.modules
        finder_ready = False
        try:
            _ensure_remote_vae_finder(snapshot)
            finder_ready = True
            remote_module = importlib.import_module(qualified_module_name)
            remote_class = getattr(remote_module, class_name)
            if not isinstance(remote_class, type):
                raise TypeError(
                    f"VAE auto_map target is not a class: {class_reference!r}"
                )
        except Exception:
            if finder_ready and package_created:
                _remove_remote_vae_package(package_name)
            raise

    return remote_class
