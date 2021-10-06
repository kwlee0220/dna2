from setuptools import setup, find_packages

setup( 
    name = 'dna',
    version = '0.0.1',
    description = 'DNA framework',
    author = 'Kang-Woo Lee',
    author_email = 'kwlee@etri.re.kr',
    url = 'https://github.com/kwlee0220/dna',
    install_requires = [
        'opencv-python',
        'opencv-contrib-python',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'requests',
        'pyyaml',
        'tqdm',
    ],
    packages = find_packages(),
    python_requires = '>=3',
    zip_safe = False
)