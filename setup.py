#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'Keras>=2.1.6',
    'numpy>=1.14.3',
    'scikit-image>=0.13.1',
    'scikit-learn>=0.19.1',
    'tensorflow>=1.8.0',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="William Xue",
    author_email='wgxue@mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Pipelines and primitives for machine learning and data science.",
    entry_points={
        'console_scripts': [
            'mlblocks=mlblocks.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='mlblocks',
    name='mlblocks',
    packages=find_packages(include=['mlblocks']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/HDI-Project/mlblocks',
    version='0.1.0',
    zip_safe=False,
)
