from setuptools import setup, find_packages

setup(
    name='Stim-PyPer',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    author='Calvin W. Howard',
    author_email='choward12@bwh.harvard.edu',
    description='A package for VTA optimization and related utilities.',
    url='https://github.com/CalvinWHow/caduceus',
    license='MIT',
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)