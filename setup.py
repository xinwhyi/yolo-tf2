from setuptools import find_packages, setup

install_requires = open('requirements.txt').read().splitlines()
install_requires.append('ml-utils@git+https://git@github.com/unsignedrant/ml-utils.git')

setup(
    name='yolo_tf2',
    version='1.0',
    packages=find_packages(),
    url='https://github.com/unsignedrant/yolo-tf2',
    license='MIT',
    author='unsignedrant',
    description='yolo(all versions) implementation in tensorflow 2.x',
    install_requires=install_requires,
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'yolotf2=yolo_tf2.cli:execute',
        ],
    },
)
