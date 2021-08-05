from setuptools import find_packages, setup

install_requires = [dep.strip() for dep in open('requirements.txt')]

setup(
    name='yolo_tf2',
    version='1.5',
    packages=find_packages(),
    url='https://github.com/schissmantics/yolov3-keras-tf2',
    license='MIT',
    author='schismantics',
    author_email='schissmantics@outlook.com',
    description='yolo(v3/v4) implementation in keras and tensorflow 2.5',
    setup_requires=['numpy==1.19.5'],
    install_requires=install_requires,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'yolotf2=yolo_tf2.cli:execute',
        ],
    },
)
