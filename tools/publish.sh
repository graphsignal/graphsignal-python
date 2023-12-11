#!/bin/bash

set -e

pandoc --from=markdown --to=rst --output=README.rst 'README.md'
tools/run-tests.sh

source venv/bin/activate

rm -f dist/*.tar.gz
python setup.py sdist bdist_wheel

for bundle in dist/*.tar.gz; do
	echo "Publishing $bundle..."
	twine check $bundle
	twine upload --repository pypi $bundle
done

deactivate
