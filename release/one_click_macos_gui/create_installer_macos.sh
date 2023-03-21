#!bash

# Initial cleanup
rm -rf dist
rm -rf build
FILE=OmicLearn.pkg
if test -f "$FILE"; then
  rm OmicLearn.pkg
fi
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n omiclearninstaller python=3.8 -y
conda activate omiclearninstaller

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_macos_gui
pip install "../../dist/OmicLearn-1.3-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.10
pyinstaller ../pyinstaller/OmicLearn.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../OmicLearn/data/*.fasta dist/OmicLearn/data

# Wrapping the pyinstaller folder in a .pkg package
mkdir -p dist/OmicLearn/Contents/Resources
cp ../logos/omiclearn_logo.icns dist/OmicLearn/Contents/Resources
mv dist/omiclearn_gui dist/OmicLearn/Contents/MacOS
cp Info.plist dist/OmicLearn/Contents
cp omiclearn_terminal dist/OmicLearn/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../logos/omiclearn_logo.png Resources/omiclearn_logo.png
chmod 777 scripts/*

pkgbuild --root dist/OmicLearn --identifier de.mpg.biochem.OmicLearn.app --version 0.3.0 --install-location /Applications/OmicLearn.app --scripts scripts OmicLearn.pkg
productbuild --distribution distribution.xml --resources Resources --package-path OmicLearn.pkg dist/omiclearn_gui_installer_macos.pkg
