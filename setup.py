from setuptools import setup, find_packages

setup(
    name='pytorch-lit',
    version='0.1.7',
    description='Lite Inference Toolkit(LIT) for PyTorch',
    license='MIT',
    packages=find_packages(),
    author='Amin Rezaei',
    author_email='AminRezaei0x443@gmail.com',
    keywords=['pytorch-lit', 'lit', 'lite-inference-toolkit', 'pytorch'],
    url='https://github.com/AminRezaei0x443/PyTorch-LIT',
    install_requires=['torch', 'numpy', 'tqdm']
)
