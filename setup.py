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
    'mock>=2.0.0',
    'pytest>=3.4.2',
]


setup_requires = [
    'pytest-runner>=2.11.1',
]


development_requires = [
    'Sphinx>=1.7.1',
    'bumpversion>=0.5.3',
    'coverage>=4.5.1',
    'flake8>=3.5.0',
    'graphviz==0.9',
    'ipython==6.5.0',
    'isort>=4.3.4',
    'matplotlib==2.2.3',
    'recommonmark>=0.4.0',
    'sphinx_rtd_theme>=0.2.4',
    'tox>=2.9.1',
    'twine>=1.10.0',
    'wheel>=0.30.0',
    'autoflake>=1.2',  # keep this at the end to avoid
    'autopep8>=1.3.5', # version incompatibilities with flake8
]


demo_requires = [
    'mlprimitives==0.1.1',
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
        'demo': demo_requires,
        'dev': demo_requires + development_requires + tests_require,
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
    version='0.2.1',
    zip_safe=False,
)
