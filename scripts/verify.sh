#!/bin/bash

rm -r  kt_refactored
git clone https://github.com/WatsonYang1999/kt_refactored.git
rm -r EduData
git clone https://github.com/WatsonYang1999/EduData.git
pip install -e EduData
ls
edudata download assistment-2009-2010-skill


ls
mkdir -p /content/kt_refactored/dataset/assist2009/
ls /content/
cp /content/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv /content/kt_refactored/dataset/assist2009/

cd /content/kt_refactored

export PYTHONPATH=$PYTHONPATH:/content/kt_refactored/kt

git fetch origin && git reset --hard origin/master
python main.py