# a3c pong

Run the following commands to run the code (Tested on macOS High Sierra Version 10.13.1):

```
1. Create a dedicated folder
mkdir a3c_pong_main_dir
cd a3c_pong_main_dir

2. Install Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sudo bash Miniconda3-latest-MacOSX-x86_64.sh

3. Clone this repo, install dependencies from conda environment.yml file
git clone git@github.com:quanvuong/a3c_pong.git
cd a3c_pong
conda create --file environment.yml --prefix pytorch2_cpu_py36
source activate pytorch2_cpu_py36

4. Install openAI gym ATARI environment
git clone git@github.com:openai/gym.git
cd gym
pip install -e '.[atari]'

5. Run
cd to a3c_pong folder path
python main.py

4. Afterwards, to undo the previous step:
rm -rf ~/miniconda3
rm -rf a3c_pong_main_dir
```

Alternatively, `sbatch.script` is the SLURM job submission script. SLURM is a job scheduler for High Performance Computing cluster. I have written a short guide on SLURM [here](https://github.com/quanvuong/deep_learning_tips_and_tricks#slurm).