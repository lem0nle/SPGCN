from setuptools import setup, find_packages
setup(
    name="comp-invest",
    version="0.1",
    packages=find_packages(),
    install_requires=['dgl', 'torch', 'numpy', 'pandas', 'tqdm', 'scikit-learn'],
)
