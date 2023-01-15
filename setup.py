"""Setup script.

Usage examples:

    pip install -e .
    pip install -e .[develop]
"""
from setuptools import find_packages, setup

setup(name="llmclone",
      version="0.0.0",
      packages=find_packages(include=["llmclone", "llmclone.*"]),
      install_requires=[
          "PyYAML",
          "types-PyYAML",
          "numpy",
          "pandas",
          "pandas-stubs",
          "openai",
          "matplotlib",
          "pyperplan",
          "transformers",
          "sentence-transformers",
      ],
      setup_requires=['setuptools_scm'],
      include_package_data=True,
      extras_require={
          "develop": [
              "mypy", "pytest-cov>=2.12.1", "pytest-pylint>=0.18.0",
              "yapf==0.32.0", "docformatter", "isort"
          ]
      })
