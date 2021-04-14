#!/bin/bash

set -e

pandoc --from=markdown --to=rst --output=README.rst 'README.md'

scripts/run-tests.sh

rm -f dist/*.tar.gz
python setup.py sdist

for bundle in dist/*.tar.gz; do
	echo "Publishing $bundle..."
	twine check $bundle
	twine upload $bundle
done

