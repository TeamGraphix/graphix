from setuptools import setup

requirements = [
    "numpy>=1.22.0",
    "networkx>=2.0",
    "qiskit>=0.20",
    "z3-solver"
]

info = {
    'name': 'Graphix',
    'version': '0.0.1',
    'packages': ['graphix'],
    'description': 'Open source library to optimize and simulate measurement-based quantum computation',
    'install_requires': requirements,
}

setup(**(info))
