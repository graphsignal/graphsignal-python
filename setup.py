import os
from setuptools import setup, find_packages

def read(fname):
  return open(os.path.join(os.path.dirname(__file__), fname)).read()

version = {}
with open('graphsignal/version.py') as fp:
    exec(fp.read(), version)

setup(
  name = 'graphsignal',
  version = version['__version__'],
  description = 'Graphsignal Profiler',
  long_description = read('README.md'),
  long_description_content_type = 'text/markdown',
  author = 'Graphsignal, Inc.',
  author_email = 'devops@graphsignal.com',
  url = 'https://graphsignal.com',
  license = 'Apache-2.0',
  keywords = [
    'machine learning',
    'deep learning',
    'data science',
    'MLOps',
    'machine learning profiler',
    'deep learning profiler',
    'machine learning performance',
    'training profiling',
    'inference profiling'
  ],
  classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Web Environment',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Topic :: Software Development',
    'Topic :: System :: Monitoring',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence'
  ],
  python_requires='>=3.5',
  install_requires=[
    'protobuf>3.0'
  ],
  
  packages = find_packages(exclude=[
    '*.sh', '*_test.py'])
)
