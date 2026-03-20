#!/bin/bash
set -e


echo "Running fig-9.sh..."
bash ./scripts/fig-9.sh
echo "--------------------------------"


echo "Running tab-2.sh..."
bash ./scripts/tab-2.sh
echo "--------------------------------"


echo "Running tab-3-left.sh..."
bash ./scripts/tab-3-left.sh
echo "--------------------------------"


echo "Running tab-3-right.sh..."
bash ./scripts/tab-3-right.sh
echo "--------------------------------"


echo "Running fig-11-acc.sh..."
bash ./scripts/fig-11-acc.sh
echo "--------------------------------"


echo "Quick run completed!"


