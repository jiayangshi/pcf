from setuptools import setup

setup(python_requires='>=3.7',
      name='pcf',
      version='0.0.1',
      description='The common used function with pytorch',
      url='https://github.com/jiayangshi/pcf',
      author='Jiayang Shi',
      author_email='j.shi@liacs.leidenuniv.nl',
      license='MIT',
      packages=['pcf'],
      install_requires=[
          'pytorch >= 1.8.0',
          'numpy',
          'matplotlib',
          'PIL'
      ],
      zip_safe=False)
