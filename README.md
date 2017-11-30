# a3c to play pong

The main file entry is `main.py`. The two main dependencies are Pytorch and OpenAI gym. Check out [environment.yml] for all dependencies and their version.

Run the following commands to run the code (Tested on factory installation of macOS Sierra Version 10.12.6 MacBook Air 13-inch, Mid 2013):

```
1. Create a dedicated folder
mkdir a3c_pong_main_dir
cd a3c_pong_main_dir

2. Install Miniconda
curl -o Miniconda3-latest-MacOSX-x86_64.sh 'https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh'
sudo bash Miniconda3-latest-MacOSX-x86_64.sh
Follow the instructions and accept the default options

3. Clone this repo, install dependencies
git clone https://github.com/quanvuong/a3c_pong.git
cd a3c_pong
sudo conda create --prefix pytorch2_cpu_py36 python=3.6 pytorch=0.2.0 -c soumith
source activate pytorch2_cpu_py36 # Activate conda environment
sudo pip install namedlist==1.7

4. Install openAI gym ATARI environment
git clone https://github.com/openai/gym.git
cd gym
git checkout 0eccce8146d011f7691f4aff5766be7c5a4d8cbd
sudo pip install -e '.[atari]'

5. Run
cd to a3c_pong folder path
python main.py

6. Afterwards, to undo the previous step:
source deactivate # Deactivate conda environment
sudo rm -rf ~/miniconda3 # Be sure you want to remove your existing conda installation
sudo rm -rf a3c_pong_main_dir
Remove the Miniconda3 install location from PATH in your .bash_profile
```

Alternatively, `sbatch.script` is the SLURM job submission script. SLURM is a job scheduler for High Performance Computing cluster. I have written a short beginners-friendly guide on SLURM [here](https://github.com/quanvuong/deep_learning_tips_and_tricks#slurm).
