#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup


with open('README.md') as readme_file:
    readme = readme_file.read()


with open('HISTORY.md') as history_file:
    history = history_file.read()


install_requires = [
]


tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'mlprimitives>=0.2,<0.3',
    'urllib3>=1.20,<1.25',
    'setuptools>=41.0.0',
    'numpy<1.17',
]


setup_requires = [
    'pytest-runner>=2.11.1',
]


development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',
    'graphviz>=0.9',
    'ipython>=6.5.0',
    'matplotlib>=2.2.3',
    'autodocsumm>=0.1.10',
    'docutils<0.15,>=0.10',    # botocore incompatibility with 0.15

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.2',  # keep this after flake8 to avoid
    'autopep8>=1.3.5', # version incompatibilities with flake8

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'tox>=2.9.1',
    'coverage>=4.5.1',

    # Documentation style
    'doc8>=0.8.0',
    'pydocstyle>=3.0.0'
]


setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Pipelines and primitives for machine learning and data science.",
    extras_require={
        'dev': development_requires + tests_require,
        'test': tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='auto machine learning classification regression data science pipeline',
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='mlblocks',
    packages=find_packages(include=['mlblocks', 'mlblocks.*']),
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-Project/MLBlocks',
    version='0.3.2',
    zip_safe=False,
)
