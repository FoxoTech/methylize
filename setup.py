# Lib
from setuptools import setup, find_packages
exec(open('methylize/version.py').read())

setup(
    name='methylize',
    version=__version__,
    description='EWAS Analysis software for Illumina methylation arrays',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    project_urls = {
        "Documentation": "https://life-epigenetics-methylize.readthedocs-hosted.com/en/latest/",
        "Source": "https://github.com/FOXOBioScience/methylize/",
        "Funding": "https://FOXOBioScience.com/"
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
      ],
    keywords='analysis methylation dna data processing life epigenetics illumina parallelization',
    url='https://github.com/FOXOBioScience/methylize',
    license='MIT',
    author='FOXO Bioscience',
    author_email='info@FOXOBioScience.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'matplotlib',
        'methylprep'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'methylize = methylize:main',
        ],
    },
)
