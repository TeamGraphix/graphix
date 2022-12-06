from setuptools import setup

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

requirements = [
    "numpy>=1.22.0",
    "networkx>=2.0",
    "z3-solver",
    "tensornetwork>=0.4"
]

info = {
    'name': 'graphix',
    'version': '0.1.0',
    'packages': ['graphix'],
    'author': 'Shinichi Sunami',
    'author_email': 'shinichi.sunami@gmail.com',
    "maintainer": "Shinichi Sunami",
    "maintainer_email": "shinichi.sunami@gmail.com",
    "license": "Apache License 2.0",
    'description': 'optimize and simulate measurement-based quantum computation',
    'long_description' : long_description,
    'long_description_content_type' : 'text/markdown',
    'url' : "https://graphix.readthedocs.io",
    'project_urls': {
        "Bug Tracker": "https://github.com/TeamGraphix/graphix/issues",
    },
    'classifiers': [
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    'python_requires': ">=3.7",
    'install_requires': requirements,
}

setup(**(info))
