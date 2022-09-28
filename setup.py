from setuptools import setup

requirements = [
    "numpy>=1.22.0",
    "networkx>=2.0",
    "qiskit>=0.20",
    "z3-solver"
]

info = {
    'name': 'Graphix',
    'version': '0.0.0',
    'packages': ['graphix'],
    'description': 'Open source library for graph-based one-way quantum computation',
    'install_requires': requirements,
}

setup(**(info))
