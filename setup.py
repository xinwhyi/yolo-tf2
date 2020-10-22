from setuptools import setup, find_packages

install_requires = [
    'imagesize==1.2.0',
    'numpy==1.18.5',
    'pandas==1.1.3',
    'seaborn==0.11.0',
    'tensorflow==2.3.1',
    'matplotlib==3.3.2',
    'lxml==4.6.1',
    'imgaug==0.4.0',
    'tensorflow_addons==0.11.2',
    'opencv_python_headless==4.4.0.44',
    'imagecorruptions==1.1.0',
    'configparser~=5.0.1',
    'scipy==1.4.1',
    'PyQt5==5.15.1'
]

setup(
    name='yolo_tf2',
    version='1.0',
    packages=find_packages(),
    url='https://github.com/emadboctorx/yolov3-keras-tf2',
    license='MIT',
    author='emadboctor',
    author_email='emad_1989@hotmail.com',
    description='yolo(v3/v4) implementation in keras and tensorflow 2.3',
    install_requires=install_requires,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'yolotf2=yolo_tf2.cli:execute',
        ],
    },
)
