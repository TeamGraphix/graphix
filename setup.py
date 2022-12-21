from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = {}
with open("graphix/version.py") as fp:
    exec(fp.read(), version)

requirements = [requirement.strip() for requirement in open("requirements.txt").readlines()]

info = {
    "name": "graphix",
    "version": version["__version__"],
    "packages": find_packages(),
    "author": "Shinichi Sunami",
    "author_email": "shinichi.sunami@gmail.com",
    "maintainer": "Shinichi Sunami",
    "maintainer_email": "shinichi.sunami@gmail.com",
    "license": "Apache License 2.0",
    "description": "Optimize and simulate measurement-based quantum computation",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "url": "https://graphix.readthedocs.io",
    "project_urls": {"Bug Tracker": "https://github.com/TeamGraphix/graphix/issues"},
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    "python_requires": ">=3.8,<3.11",
    "install_requires": requirements,
}

setup(**(info))
