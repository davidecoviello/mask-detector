#!/bin/zsh
set -e
echo "Creating kernel computer-vision ..."
python3 -m venv venv
. venv/bin/activate
ipython kernel install --name "computer-vision" --user
echo "Done!"
set +e
