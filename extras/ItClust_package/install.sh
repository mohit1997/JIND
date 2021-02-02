conda create -n ItClust python=3.6
conda activate ItClust
conda install -c conda-forge tensorflow
pip install tensorflow --upgrade
pip install ItClust
pip install tensorflow==1.14 --force
pip install keras==2.2.4 --force
pip install scanpy==1.5.1 --force
