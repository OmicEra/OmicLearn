import setuptools

import omiclearn as package2install


def get_long_description():
    with open("README.md", "r") as readme_file:
        long_description = readme_file.read()
    return long_description


with open("reqs.txt") as f:
    required = f.read().splitlines()


def create_pip_wheel():
    requirements = required
    setuptools.setup(
        name=package2install.__project__,
        version=package2install.__version__,
        license=package2install.__license__,
        description=package2install.__description__,
        author=package2install.__author__,
        author_email=package2install.__author_email__,
        url=package2install.__github__,
        project_urls=package2install.__urls__,
        keywords=package2install.__keywords__,
        classifiers=package2install.__classifiers__,
        packages=setuptools.find_packages(),
        include_package_data=True,
        entry_points={
            "console_scripts": package2install.__console_scripts__,
        },
        install_requires=requirements,
        python_requires=package2install.__python_version__,
    )


if __name__ == "__main__":
    create_pip_wheel()
