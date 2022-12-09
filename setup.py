#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()


install_requires = [
    'graphviz>=0.9,<1',
    'numpy>=1.17.1,<2',
    'psutil>=5,<6',
]


mlprimitives_requires = [
    'mlprimitives>=0.3.0,<0.4',
    'h5py<2.11.0,>=2.10.0',  # <- tensorflow 2.3.2 conflict
    'matplotlib<3.2.2,>=2.2.2',  # <- copulas 0.3.3
    'protobuf<4', # <- importlib
]

examples_require = mlprimitives_requires + [
    'jupyter==1.0.0',
    'baytune>=0.4.0,<0.5',
]


tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'setuptools>=41.0.0',
    'rundoc>=0.4.3',
    'prompt-toolkit>=2.0,<3.0',
]


setup_requires = [
    'pytest-runner>=2.11.1',
]


development_requires = [
    # general
    'bumpversion>=0.5.3,<0.6',
    'pip>=9.0.1',
    'watchdog>=0.8.3,<0.11',

    # docs
    'm2r>=0.2.0,<0.3',
    'Sphinx>=1.7.1,<3',
    'sphinx_rtd_theme>=0.2.4,<0.5',
    'docutils>=0.12,<0.18',
    'ipython>=6.5.0',
    'autodocsumm>=0.1.10',
    'Jinja2>=2,<3', # >=3 makes sphinx theme fail
    'markupsafe<2.1.0',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<2',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',

    # Documentation style
    'doc8>=0.8.0',
    'pydocstyle>=3.0.0',
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='Pipelines and primitives for machine learning and data science.',
    extras_require={
        'dev': development_requires + tests_require + examples_require,
        'unit': tests_require,
        'test': tests_require + examples_require,
        'examples': examples_require,
        'mlprimitives': mlprimitives_requires,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='auto machine learning classification regression data science pipeline',
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='mlblocks',
    packages=find_packages(include=['mlblocks', 'mlblocks.*']),
    python_requires='>=3.6,<3.9',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/MLBazaar/MLBlocks',
    version='0.4.2.dev0',
    zip_safe=False,
)
