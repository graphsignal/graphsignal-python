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
  description = 'Graphsignal Logger',
  long_description = read('README.md'),
  long_description_content_type = 'text/markdown',
  author = 'Graphsignal, Inc.',
  author_email = 'devops@graphsignal.ai',
  url = 'https://graphsignal.ai',
  license = 'BSD',
  keywords = [
    'machine learning',
    'deep learning',
    'data science',
    'MLOps',
    'machine learning devops',
    'machine learning monitoring',
    'ML monitoring',
    'AI monitoring',
    'pipeline monitoring',
    'model monitoring',
    'prediction monitoring',
    'data drift',
    'concept drift',
    'training-serving skew'
  ],
  classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Web Environment',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: System Administrators',
    'License :: OSI Approved :: BSD License',
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
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn'
  ],
  
  packages = find_packages(exclude=[
    '*.sh', '*_test.py', 'examples'])
)
