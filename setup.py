from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(python_requires='>=3.7',
    name='pcfv',
    version='0.0.4',
    description='The common used function with pytorch for vision tasks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
    'Development Status :: 1 - Planning',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    url='https://github.com/jiayangshi/pcf',
    author='Jiayang Shi',
    author_email='j.shi@liacs.leidenuniv.nl',
    license='MIT',
    packages=['pcfv'],
    install_requires=[
        'pytorch',
        'numpy',
        'matplotlib',
        'Pillow'
    ],
    zip_safe=False)
