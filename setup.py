from setuptools import find_packages, setup

install_requires = [
    'imagesize==1.3.0',
    'imgaug==0.4.0',
    'lxml==4.6.4',
    'matplotlib==3.4.3',
    'numpy==1.19.5',
    'opencv-python-headless==4.5.4.58',
    'pandas==1.3.4',
    'seaborn==0.11.2',
    'setuptools==58.5.3',
    'tensorflow==2.7.0',
    'tensorflow_addons==0.15.0',
    'imagecorruptions==1.1.2',
    'configparser~=5.1.0',
    'scipy==1.7.1',
    'PyQt5==5.15.4',
    'tabulate==0.8.9',
    'ipykernel==6.4.2',
]

setup(
    name='yolo_tf2',
    version='1.5',
    packages=find_packages(),
    url='https://github.com/schissmantics/yolo-tf2',
    license='MIT',
    author='schismantics',
    author_email='schissmantics@outlook.com',
    description='yolo(v3/v4) implementation in keras and tensorflow 2.7',
    setup_requires=['numpy==1.19.5'],
    install_requires=install_requires,
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'yolotf2=yolo_tf2.cli:execute',
        ],
    },
)
