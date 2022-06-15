import os
import sys
sys.path.append(os.path.join(os.getcwd(), r'resources\models'))
sys.path.append(os.path.join(os.getcwd(), r'resources\models\research'))
sys.path.append(os.path.join(os.getcwd(), r'resources\models\research\slim'))
os.environ['PYTHONPATH'] += ';' + os.path.join(os.getcwd(), r'resources\models\research;')
os.environ['PYTHONPATH'] += os.path.join(os.getcwd(), r'resources\models;')
os.environ['PYTHONPATH'] += os.path.join(os.getcwd(), r'resources\models\research\slim;')

import PrintUtils
from colorama import Fore as Color
from ProjectFinal.project_gui import ProjectGui

def main():
    if 'CONDA_DEFAULT_ENV' in os.environ:
        env = os.environ['CONDA_DEFAULT_ENV']
    else:
        env = 'Default Python'
    python_version = ".".join(map(str, sys.version_info[0:3]))

    PrintUtils.info('Current conda environment: {}'.format(Color.LIGHTGREEN_EX + env))
    PrintUtils.info('Current python version: {}'.format(Color.LIGHTGREEN_EX + python_version))

    projectgui = ProjectGui()
    projectgui.main_window()


if __name__ == '__main__':
    main()
