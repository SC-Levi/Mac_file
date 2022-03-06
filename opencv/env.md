1. anaconda create env
/###2. conda install pytorch torchvision(pip install -r requirements.txt) 1.7.1   0.8.0/0.8.1
1) gpu  cuda cudnn 显卡
2) cpu  cuda/
3. cuda cpu 
<teamviewer>
sudo apt-get update
sudo apt-get upgrade

4. pip version
pip install -r requirements.txt
install pip3
sudo apt-get install python3-pip3 python3-dev
python3 -m pip install --upgrade pip

sudo gedit /usr/bin/pip3
将原来的

from pip import main

if __name__ == '__main__':

    sys.exit(main())

改成

from pip import __main__

if __name__ == '__main__':

    sys.exit(__main__._main())

pip3 -V


pip install -r requirements.txt




