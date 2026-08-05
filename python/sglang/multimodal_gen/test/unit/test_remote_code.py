import sys
import tempfile
import textwrap
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from sglang.multimodal_gen.runtime.loader.component_loaders import remote_code
from sglang.multimodal_gen.runtime.loader.component_loaders.remote_code import (
    _load_remote_vae_class,
    _remote_vae_package_name,
)


def _write_remote_module(root: Path, relative_path: str, source: str) -> None:
    module_path = root / relative_path
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


class TestRemoteVAECodeLoader(unittest.TestCase):
    def setUp(self):
        self._package_names: set[str] = set()

    def tearDown(self):
        module_names = {
            name
            for name in sys.modules
            if any(
                name == package_name or name.startswith(f"{package_name}.")
                for package_name in self._package_names
            )
        }
        remote_code._remove_remote_vae_modules(module_names)

    def _track_package(self, component_root: Path) -> None:
        self._package_names.add(_remote_vae_package_name(component_root.resolve()))

    def test_remote_class_supports_relative_nested_and_circular_imports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            component_root = Path(temp_dir)
            _write_remote_module(component_root, "sibling.py", 'TOKEN = "relative"\n')
            _write_remote_module(component_root, "nested/__init__.py", "")
            _write_remote_module(
                component_root, "nested/value.py", 'TOKEN = "nested"\n'
            )
            _write_remote_module(
                component_root,
                "cycle.py",
                """
                from . import modeling_vae

                def entry_module_name():
                    return modeling_vae.__name__
                """,
            )
            _write_remote_module(
                component_root,
                "modeling_vae.py",
                """
                from . import cycle
                from .nested.value import TOKEN as NESTED_TOKEN
                from .sibling import TOKEN

                class RemoteVAE:
                    marker = (TOKEN, NESTED_TOKEN, cycle.entry_module_name())
                """,
            )
            self._track_package(component_root)

            remote_class = _load_remote_vae_class(
                str(component_root),
                "modeling_vae.RemoteVAE",
                trust_remote_code=True,
            )

            self.assertEqual(remote_class.marker[:2], ("relative", "nested"))
            self.assertEqual(remote_class.marker[2], remote_class.__module__)

    def test_remote_class_rejects_untrusted_code_before_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            component_root = Path(temp_dir)
            sentinel = component_root / "executed.txt"
            _write_remote_module(
                component_root,
                "modeling_vae.py",
                f"""
                from pathlib import Path

                Path({str(sentinel)!r}).write_text("executed", encoding="utf-8")

                class RemoteVAE:
                    pass
                """,
            )
            self._track_package(component_root)

            with self.assertRaisesRegex(ValueError, "trust_remote_code=True"):
                _load_remote_vae_class(
                    str(component_root),
                    "modeling_vae.RemoteVAE",
                    trust_remote_code=False,
                )

            self.assertFalse(sentinel.exists())

    def test_untrusted_code_is_rejected_before_source_scan(self):
        with patch.object(
            remote_code, "_snapshot_remote_vae_sources"
        ) as snapshot_sources:
            with self.assertRaisesRegex(ValueError, "trust_remote_code=True"):
                _load_remote_vae_class(
                    "/path/that/does/not/exist",
                    "modeling_vae.RemoteVAE",
                    trust_remote_code=False,
                )

        snapshot_sources.assert_not_called()

    def test_external_file_symlink_lazy_namespace_import_uses_snapshot(self):
        with (
            tempfile.TemporaryDirectory() as component_temp_dir,
            tempfile.TemporaryDirectory() as source_temp_dir,
        ):
            component_root = Path(component_temp_dir)
            source_root = Path(source_temp_dir)
            lazy_directory = component_root / "lazy_namespace"
            lazy_directory.mkdir()
            target = source_root / "external_value.py"
            target.write_text('TOKEN = "snapshotted"\n', encoding="utf-8")
            (lazy_directory / "value.py").symlink_to(target)
            _write_remote_module(
                component_root,
                "modeling_vae.py",
                """
                class RemoteVAE:
                    @classmethod
                    def lazy_marker(cls):
                        from .lazy_namespace.value import TOKEN

                        return TOKEN
                """,
            )
            original_package_name = _remote_vae_package_name(component_root)
            self._package_names.add(original_package_name)
            original_sys_path = list(sys.path)

            remote_class = _load_remote_vae_class(
                str(component_root),
                "modeling_vae.RemoteVAE",
                trust_remote_code=True,
            )
            target.write_text('TOKEN = "changed-on-disk"\n', encoding="utf-8")
            changed_package_name = _remote_vae_package_name(component_root)
            self._package_names.add(changed_package_name)

            self.assertEqual(remote_class.lazy_marker(), "snapshotted")
            self.assertEqual(sys.path, original_sys_path)
            self.assertNotEqual(changed_package_name, original_package_name)

            changed_remote_class = _load_remote_vae_class(
                str(component_root),
                "modeling_vae.RemoteVAE",
                trust_remote_code=True,
            )
            self.assertEqual(changed_remote_class.lazy_marker(), "changed-on-disk")
            self.assertEqual(remote_class.lazy_marker(), "snapshotted")
            self.assertIsNot(changed_remote_class, remote_class)

    def test_namespace_topology_is_part_of_package_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            component_root = Path(temp_dir)
            _write_remote_module(
                component_root,
                "modeling_vae.py",
                """
                class RemoteVAE:
                    pass
                """,
            )
            package_without_namespace = _remote_vae_package_name(component_root)
            (component_root / "empty_namespace").mkdir()
            package_with_namespace = _remote_vae_package_name(component_root)

            self.assertNotEqual(package_without_namespace, package_with_namespace)

    def test_directory_symlink_is_rejected_before_execution(self):
        with (
            tempfile.TemporaryDirectory() as component_temp_dir,
            tempfile.TemporaryDirectory() as target_temp_dir,
        ):
            component_root = Path(component_temp_dir)
            sentinel = component_root / "executed.txt"
            _write_remote_module(
                component_root,
                "modeling_vae.py",
                f"""
                from pathlib import Path

                Path({str(sentinel)!r}).write_text("executed", encoding="utf-8")

                class RemoteVAE:
                    pass
                """,
            )
            (component_root / "linked_directory").symlink_to(
                target_temp_dir, target_is_directory=True
            )

            with self.assertRaisesRegex(ValueError, "directory symlinks"):
                _load_remote_vae_class(
                    str(component_root),
                    "modeling_vae.RemoteVAE",
                    trust_remote_code=True,
                )

            self.assertFalse(sentinel.exists())

    def test_broken_and_cyclic_python_symlinks_are_rejected(self):
        for link_kind in ("broken", "cyclic"):
            with (
                self.subTest(link_kind=link_kind),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                component_root = Path(temp_dir)
                _write_remote_module(
                    component_root,
                    "modeling_vae.py",
                    """
                    class RemoteVAE:
                        pass
                    """,
                )
                if link_kind == "broken":
                    (component_root / "broken.py").symlink_to("missing.py")
                else:
                    (component_root / "first.py").symlink_to("second.py")
                    (component_root / "second.py").symlink_to("first.py")

                with self.assertRaisesRegex(ValueError, "broken or cyclic"):
                    _load_remote_vae_class(
                        str(component_root),
                        "modeling_vae.RemoteVAE",
                        trust_remote_code=True,
                    )

    def test_invalid_auto_map_references_are_rejected(self):
        invalid_references = (
            None,
            "RemoteVAE",
            "../modeling.RemoteVAE",
            "modeling..RemoteVAE",
            "modeling.class",
            ["modeling.RemoteVAE"],
        )
        for class_reference in invalid_references:
            with self.subTest(class_reference=class_reference):
                with self.assertRaisesRegex(ValueError, "auto_map"):
                    _load_remote_vae_class(
                        "/path/that/does/not/exist",
                        class_reference,
                        trust_remote_code=True,
                    )

    def test_malformed_auto_map_container_is_rejected(self):
        for auto_map in (None, [], "modeling_vae.RemoteVAE"):
            with (
                self.subTest(auto_map=auto_map),
                self.assertRaisesRegex(ValueError, "auto_map must be a mapping"),
            ):
                remote_code._get_remote_vae_auto_model_reference({"auto_map": auto_map})

    def test_same_named_remote_modules_are_isolated_by_component_root(self):
        with (
            tempfile.TemporaryDirectory() as first_temp_dir,
            tempfile.TemporaryDirectory() as second_temp_dir,
        ):
            first_root = Path(first_temp_dir)
            second_root = Path(second_temp_dir)
            for component_root, token in (
                (first_root, "first"),
                (second_root, "second"),
            ):
                _write_remote_module(
                    component_root, "sibling.py", f"TOKEN = {token!r}\n"
                )
                _write_remote_module(
                    component_root,
                    "modeling_vae.py",
                    """
                    from .sibling import TOKEN

                    class RemoteVAE:
                        marker = TOKEN
                    """,
                )
                self._track_package(component_root)

            first_class = _load_remote_vae_class(
                str(first_root),
                "modeling_vae.RemoteVAE",
                trust_remote_code=True,
            )
            second_class = _load_remote_vae_class(
                str(second_root),
                "modeling_vae.RemoteVAE",
                trust_remote_code=True,
            )
            first_class_again = _load_remote_vae_class(
                str(first_root),
                "modeling_vae.RemoteVAE",
                trust_remote_code=True,
            )

            self.assertEqual(first_class.marker, "first")
            self.assertEqual(second_class.marker, "second")
            self.assertIs(first_class_again, first_class)
            self.assertNotEqual(first_class.__module__, second_class.__module__)

    def test_failed_remote_import_is_cleaned_before_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            component_root = Path(temp_dir)
            _write_remote_module(
                component_root,
                "modeling_vae.py",
                """
                from .sibling import TOKEN

                class RemoteVAE:
                    marker = TOKEN
                """,
            )
            failed_package_name = _remote_vae_package_name(component_root.resolve())
            self._package_names.add(failed_package_name)

            with self.assertRaises(ModuleNotFoundError):
                _load_remote_vae_class(
                    str(component_root),
                    "modeling_vae.RemoteVAE",
                    trust_remote_code=True,
                )

            self.assertFalse(
                any(
                    name == failed_package_name
                    or name.startswith(f"{failed_package_name}.")
                    for name in sys.modules
                )
            )

            _write_remote_module(component_root, "sibling.py", 'TOKEN = "ready"\n')
            self._track_package(component_root)
            remote_class = _load_remote_vae_class(
                str(component_root),
                "modeling_vae.RemoteVAE",
                trust_remote_code=True,
            )

            self.assertEqual(remote_class.marker, "ready")

    def test_failed_import_cleanup_does_not_remove_successful_package(self):
        with (
            tempfile.TemporaryDirectory() as successful_temp_dir,
            tempfile.TemporaryDirectory() as failing_temp_dir,
        ):
            successful_root = Path(successful_temp_dir)
            failing_root = Path(failing_temp_dir)
            _write_remote_module(
                successful_root,
                "modeling_vae.py",
                """
                class RemoteVAE:
                    marker = "successful"
                """,
            )
            _write_remote_module(
                failing_root,
                "modeling_vae.py",
                """
                from .missing import TOKEN

                class RemoteVAE:
                    marker = TOKEN
                """,
            )
            successful_package = _remote_vae_package_name(successful_root)
            failing_package = _remote_vae_package_name(failing_root)
            self._package_names.update({successful_package, failing_package})

            successful_class = _load_remote_vae_class(
                str(successful_root),
                "modeling_vae.RemoteVAE",
                trust_remote_code=True,
            )
            with self.assertRaises(ModuleNotFoundError):
                _load_remote_vae_class(
                    str(failing_root),
                    "modeling_vae.RemoteVAE",
                    trust_remote_code=True,
                )

            self.assertIn(successful_package, sys.modules)
            self.assertIn(successful_package, remote_code._REMOTE_VAE_FINDERS)
            self.assertNotIn(failing_package, sys.modules)
            self.assertNotIn(failing_package, remote_code._REMOTE_VAE_FINDERS)
            self.assertIs(
                _load_remote_vae_class(
                    str(successful_root),
                    "modeling_vae.RemoteVAE",
                    trust_remote_code=True,
                ),
                successful_class,
            )

    def test_concurrent_remote_import_executes_one_module_instance(self):
        worker_count = 16
        with tempfile.TemporaryDirectory() as temp_dir:
            component_root = Path(temp_dir)
            _write_remote_module(component_root, "sibling.py", 'TOKEN = "ready"\n')
            _write_remote_module(
                component_root,
                "modeling_vae.py",
                """
                import time

                from .sibling import TOKEN

                time.sleep(0.05)

                class RemoteVAE:
                    marker = TOKEN
                """,
            )
            self._track_package(component_root)
            barrier = threading.Barrier(worker_count)

            def load_once(_worker_index: int):
                barrier.wait(timeout=5)
                return _load_remote_vae_class(
                    str(component_root),
                    "modeling_vae.RemoteVAE",
                    trust_remote_code=True,
                )

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                remote_classes = list(executor.map(load_once, range(worker_count)))

            self.assertEqual(
                {remote_class.marker for remote_class in remote_classes}, {"ready"}
            )
            self.assertEqual(
                len({id(remote_class) for remote_class in remote_classes}), 1
            )


if __name__ == "__main__":
    unittest.main()
