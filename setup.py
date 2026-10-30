"""Build the standalone C reader; wheels have no Python C-API dependency."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


class BuildNative(build_py):
    def run(self):
        if sys.platform not in ("darwin", "linux"):
            raise RuntimeError("native row-store wheels currently support Linux and macOS")
        super().run()
        directory = Path(self.build_lib) / "meft_memory" / "_native"
        directory.mkdir(parents=True, exist_ok=True)
        darwin = sys.platform == "darwin"
        filename = "libmeft_rows.dylib" if darwin else "libmeft_rows.so"
        compiler = shlex.split(os.environ.get("CC", "cc"))
        command = [*compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC",
                   "-dynamiclib" if darwin else "-shared", str(Path(__file__).parent / "native" / "meft_rows.c"),
                   "-lz", "-o", str(directory / filename)]
        subprocess.run(command, check=True)


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(cmdclass={"build_py": BuildNative, "bdist_wheel": PlatformWheel})
