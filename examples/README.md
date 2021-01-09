# MLBlocks Examples

This folder contains Python code, Jupyter Notebooks and JSON examples to demonstrate MLBlocks
functionaliry.

Within this folder you will find:

<!--* `examples.py`: Simple Python code examples of a class and a function based primitive implementation.-->
* `primitives`: Example primitive JSONs to demonstrate different MLBlocks functionalities.
* `pipelines`: Example pipeline JSONs to demonstrate different MLBlocks functionalities.
* `tutorials`: Collection of Jupyter Notebooks to show the usage of different MLBlocks functionalities.
<!--* `problem_types`: Collection of Jupyter Notebooks that show example pipelines for multiple problem types.-->

# Requirements

In order to run the examples contained in this folder you should have [pip installed on your system
](https://pip.pypa.io/en/stable/installing/).

Optionally, also install and activate a [virtualenv](https://virtualenv.pypa.io/en/latest/) to
run them in an isolated environment.

# Usage

In order to run these tutorials on your computer, please follow these steps:

1. Clone this github repository:

```bash
git clone git@github.com:MLBazaar/MLBlocks.git
```

2. (Optional) Create a virtualenv to execute the examples in an environment isolated from the
rest of your computer:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) mlblocks-venv
soucre mlblocks-venv/bin/activate
```

3. Enter the repository and install the dependencies

```bash
cd MLBlocks
make install-examples
```

This will install [MLBLocks](https://github.com/MLBazaar/MLBlocks.git) as well as [MLPrimitives](
https://github.com/MLBazaar/MLPrimitives.git) and [Jupyter](https://jupyter.org/).

4. Enter the `examples` folder and start a Jupyter Notebook:

```bash
jupyter notebook
```

5. Point your browser at the link shown in your console and run the examples from the `examples/tutorials` folder.
