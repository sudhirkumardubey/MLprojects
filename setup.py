from setuptools import find_packages, setup # it will find all the packages that is required in machine learning project
from typing import List


HYPEN_E_DOT = '-e .' # creating a constant

def get_requirements(file_path:str)->List[str]:
    
    '''
    This function returns the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

    

setup(
    name = 'mlproject',
    version = '0.0.1',
    author= 'sudhir',
    author_email='dubey.sudhirpp@gmail.com',
    packages=find_packages(),
    # install_requires =['pandas', 'numpy', 'seaborn']
    install_requires = get_requirements('requirements.txt')
)