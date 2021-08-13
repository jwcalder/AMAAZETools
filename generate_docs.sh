sed -i 's/import amaazetools/#import amaazetools/' amaazetools/svi.py;
pdoc --html --force -o docs/ ./amaazetools;
mv docs/amaazetools/* docs/;
rmdir docs/amaazetools/;
sed -i 's/#import amaazetools/import amaazetools/' amaazetools/svi.py
