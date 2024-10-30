# setup.py
from setuptools import setup, find_packages

setup(
    name='python-face-recognition-attendance-Binus',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        'opencv-python',
        'firebase-admin',
        'face-recognition',
        # Add any other dependencies
    ],
    description='A face recognition-based attendance system for Binus',
    author='Muhammad Hardwin Variansyah',
)
