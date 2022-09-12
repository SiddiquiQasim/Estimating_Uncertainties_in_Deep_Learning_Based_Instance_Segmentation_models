from setuptools import find_packages, setup

setup(
    name='uncertainty_calculation',
    packages=find_packages(),
    version='0.1.0',
    description='descr',
    author='author',
    license='MIT',
    install_requires=[
        'tqdm==4.64.1',
        'PyYAML==6.0',
        'numpy==1.23.3',
        'scipy==1.9.0',
        'scikit-learn==1.1.2',
        'tensorflow==2.9.0',
        'matplotlib==3.5.3',
        'opencv-python==4.6.0.66',
        'stardist==0.8.3'
    ]
)