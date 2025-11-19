from setuptools import setup, find_packages

setup(
    name='bayes',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here.
        # For example: 'numpy>=1.20', 'scipy>=1.6',
    ],
    long_description=open('README.md').read(),
    python_requires='>=3.7',
)
