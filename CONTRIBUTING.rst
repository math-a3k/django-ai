.. _contributing:

============
Contributing
============

Contributions are welcome and highly appreciated!! Every little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/math-a3k/django-ai/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

django-ai could always use more documentation, whether as part of the official django-ai docs, in docstrings, or even on the web in blog posts, articles, and such. Even questions in community sites as stackoverflow generates documentation :)

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at
https://github.com/math-a3k/django-ai/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a community project, and that contributions are welcome :)

Although conversely you may use any of the communications channels - such as the mailing list - to send it, the important thing is the feedback, not the mean :)

 
Artwork
~~~~~~~

Artwork - logos, banners, themes, etc. - is highly appreciated and always welcomed.


Monetary
~~~~~~~~

You can support and ensure the `django-ai` development by making money arrive to the project in its different ways:

Donations
  Software development has costs, any help for lessen them is highly appreciated and encourages a lot to keep going :)

Sponsoring
  Hire the devs for working in a specific feature you would like to have or a bug to squash in a timely manner.

Hiring, Contracting and Consultancy
  Hire the developers to work for you (implementing code, models, etc.) in its different modalities. Even if it is not `django-ai` related, as long as the devs have enough for a living, the project will keep evolving.

Do you need to evade taxes? Lessen your profit margins by giving to `django-ai`!! Do you run an international drug cartel? Avoid money stash overflow by donating to `django-ai`!! =P


Non-Monetary
~~~~~~~~~~~~

You can support and ensure the `django-ai` development by making any good or service arrive to the project.

Anything that you consider that is helpful in the Software Development Process - as a whole - and the Proyect Sustainability is highly appreciated and encourages a lot to keep going :)


Promotion
~~~~~~~~~

Blog posts, articles, talks, etc. Anything that improves the difussion of the project is also a Contribution and helps spreading it (in a wide sense). 

Get Started!
------------

Ready to contribute code or documentation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to set up `django-ai` for local development.

1. Fork the `django-ai` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:<your_name_here>/django-ai.git

3. Install your local copy into a virtualenv. This is how you set up your fork for local development::

    $ python3 -m venv django-ai-env
    $ source django-ai-env/bin/activate
    $ cd django-ai/
    $ pip install -r requirements_dev.txt

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 reasonably (or your preferred pep8 linter) and the tests (including tox)::

        $ flake8 django_ai tests
        $ PYTHONHASHSEED=0 python runtests.py
        $ tox

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website to the ``development`` branch. Once your changes are reviewed, you may be assigned to review another pull request with improvements on your code if deemed necessary. Once we agree on a final result, it will be merged to ``master``.

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for the building matrix of CI. Check https://travis-ci.org/math-a3k/django-ai/pull_requests and make sure that the tests pass for all supported environments.

Tips
~~~~

To run a particular of test::

    $ PYTHONHASHSEED=0 python runtests.py tests.test_bns.TestDjango_ai.<test_name>

Ready to make a monetary contribution?
--------------------------------------

Contact the lead developer or use any of the communication channels and - no matter how micro it is - we will find a way of making it happen :)

Ready to make a non-monetary contribution?
------------------------------------------

Contact the lead developer or use any of the communication channels and - no matter how micro it is - we will find a way of making it happen :)

Ready to make a promotion contribution?
---------------------------------------

Contact the lead developer or use any of the communication channels and it will be listed :)

Ready to make an artwork contribution?
--------------------------------------

If you don't feel comfortable with `git`, use the GitHub wiki - https://github.com/math-a3k/django-ai/wiki - and the mailing list for submitting - django-ai@googlegroups.com.