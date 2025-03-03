from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from README.md if you have one
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''

setup(
    name='analyser-batch',
    version='0.1.0',
    author='Sobhan RJZ',
    author_email='sobhan.rajabzadeh@gmail.com',
    description='A batch analysis tool using Detectron2 for computer vision tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sobhan-rjz/Analyser_Batch',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10'
    ],
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=22.3.0',
            'isort>=5.10.1',
            'flake8>=4.0.1',
        ],
    },
    entry_points={
        'console_scripts': [
            'analyser=analyser.main:main',
            'analyser-install=install:main',
        ],
    },
    scripts=['install.py'],
) 