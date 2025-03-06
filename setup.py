#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROM: Range of Motion Assessment Tool

A tool for remote physiotherapy assessment that measures joint
range of motion from video input.
"""

import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the version from the package
version = {}
with open(os.path.join(this_directory, 'ROM', '__init__.py'), encoding='utf-8') as f:
    exec(f.read(), version)

# Main setup configuration
setup(
    name="rom-assessment",
    version=version.get('__version__', '0.1.0'),
    packages=find_packages(),
    
    # Dependencies are managed in setup.cfg
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'rom=ROM.core:main',
        ],
    },
    
    # Include non-Python files in the package
    include_package_data=True,
    package_data={
        'ROM': ['demo/*', 'demo/**/*'],
    },
    
    # Additional information
    author="Your Name",
    author_email="your.email@example.com",
    description="Range of Motion Assessment Tool for Remote Physiotherapy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ROM",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
)