from setuptools import setup, find_packages

setup(
    name='HardwiredGenome',  # Replace with your package name
    version='0.1.0',  # Initial release version
    author='Joshua Pickard and Yuchen Shao',  # Replace with your name
    author_email='jpic@umich.edu',  # Replace with your email
    description='A small package for the Hardwired Genome project',  # Short description
    long_description=open('README.md').read(),  # Long description read from the README file
    long_description_content_type='text/markdown',  # This is important if you have markdown content
    url='https://github.com/Jpickard1/HardwiredGenome-Python',  # Replace with your GitHub repository URL
    packages=find_packages(),  # Automatically find packages in your directory
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your license if different
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the Python versions you support
    install_requires=[
        # List your package dependencies here, e.g., 'numpy', 'pandas'
    ],
    extras_require={
        'dev': [
            # List additional dependencies for development, e.g., 'pytest', 'sphinx'
        ],
    },
    entry_points={
        'console_scripts': [
            # Define any console scripts here
            # Example: 'your-script = your_package.module:function',
        ],
    },
)
