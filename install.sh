sudo apt update
sudo apt install -y software-properties-common \
                    libeccodes-dev \
                    ffmpeg \
                    libsm6 \
                    libxext6


sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y  python3-pip \
                python3.8 \
                python3.8-dev \
                python3-venv \
                python3.8-venv
echo "Creating python3.8 virtual environment.."
python3.8 -m venv py38

pip install scikit-learn matplotlib pandas