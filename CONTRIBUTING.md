# Contributing

Firstly, contributions are welcome and we appreciate all of them!.

You can contribute **OmicLearn** in many ways:

## Types of Contributions

### 1. Report Bugs

Report bugs at https://github.com/MannLabs/OmicLearn/issues/new/choose by selecting `Bug report` from the list.

If you are reporting a bug/problem, please include:

-   Your operating system name and version.
-   Your web browser name with its version.
-   Python version
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug/problem you face.
-   You may also include screenshots and it might help us to understand the bug.

### 2. Fix Bugs

If you might interested in fixing bugs, look through the [`GitHub issues`](https://github.com/MannLabs/OmicLearn/issues). The issues tagged with "bug" or "help wanted" is open to whoever wants to implement it.

### 3. Request Feature / Submit Feedback

New feature requests are also welcome!
You might have a look at our `Feature request` template on [`GitHub issues`](https://github.com/MannLabs/OmicLearn/issues).


### 4. Write Documentation

Implementing our [OmicLearn Documentation](https://omiclearn.readthedocs.io/en/latest/) page would be great idea! You might add articles, detailed explanations or blog posts into the our pages.

---

## 👇 Get Started!

Ready to contribute? Here's how to set up `OmicLearn` for local development.

1.  Fork the `OmicLearn` repo on GitHub.
2.  Clone your fork locally:

    ```bash
    git clone https://github.com/MannLabs/OmicLearn.git
    ```

3.  Follow the installation instructions in the [Installation & Running](https://omiclearn.readthedocs.io/en/latest//HOW-TO:-Installation-&-Running) page for installing `OmicLearn` own environment.

4.  Create a branch for local development:

    ```bash
    git checkout -b name-of-yourbugfix-or-new-feature
    ```

    Now you can make your changes locally.

5. Format the code with `black` and `isort`

    ```bash
    black . && isort --profile black . # Default --line-length 88
    ```

6.  Commit your changes and push your branch to GitHub::

    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-yourbugfix-or-new-feature
    ```

7.  Submit a pull request through the GitHub website.

### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  You need to open a issue on Github issues page for discussing your changes.
2.  If the pull request adds functionality, the Docs should be updated.
3.  The pull request should work for Python 3.9.
4.  The functionality of project should be tested from the browser.
