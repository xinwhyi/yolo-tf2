from setuptools import find_packages, setup

install_requires = [dep.strip() for dep in open('requirements.txt')]

setup(
    name='yolo_tf2',
    version='1.4',
    packages=find_packages(),
    url='https://github.com/emadboctorx/yolov3-keras-tf2',
    license='MIT',
    author='emadboctor',
    author_email='emad_1989@hotmail.com',
    description='yolo(v3/v4) implementation in keras and tensorflow 2.3',
    setup_requires=['numpy==1.18.5'],
    install_requires=install_requires,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'yolotf2=yolo_tf2.cli:execute',
        ],
    },
)
