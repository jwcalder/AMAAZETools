#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' 's/import amaazetools/#import amaazetools/' amaazetools/svi.py;
else
	sed -i 's/import amaazetools/#import amaazetools/' amaazetools/svi.py;
fi

pdoc --html --force -o docs/ ./amaazetools;
mv docs/amaazetools/* docs/;
rmdir docs/amaazetools/;

if [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' 's/#import amaazetools/import amaazetools/' amaazetools/svi.py
else
	sed -i 's/#import amaazetools/import amaazetools/' amaazetools/svi.py
fi
