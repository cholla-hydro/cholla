This directory provides details about our Continuous Integration using GitHub actions.


# Overview

We enable multiple kinds of continuous integration to aide with the review of Cholla Pull Requests. At a high level, this machinery includes:

1. Jenkins (run on the CRC @ Pitt)
   - actually builds and tests the code (with Nvidia GPUs)
   - also performs linting

2. pre-commit.ci
   - performs code formatting (e.g. clang-format)
   - can be instructed to push a commit to the branch in a PR that fixes formatting issues.

3. GitHub Actions
   - primarily performs a compilation check to verify that Cholla can be built with AMD GPUs (but we can't actually test the code)
   - also provides a workflow to generate the container image to use in the compile-test


# More about about GitHub Actions

## Compilation Checks

In more detail, GitHub Actions uses the logic within ``compilation_checks.yml`` to actually run the compilation checks. This logic is executed every time a new PR is issued (or a new commit is added to a PR).

Importantly, all of this logic is executed within a docker image. This docker image contains all of the dependencies that we need for building Cholla 

## Building Images

Periodically, we need to bump the versions of compilers/libraries used within the docker image that we use for the compilation check.

This is where the workflow within **build_image.yml** becomes relevant.

**TODO: DESCRIBE UPDATE PROCEDURE**

## Other Thoughts

As an experiment, I tried to see if we could get away with running the Compilation Tests without constructing a custom image. In more detail, I tried to make the compilation tests run using AMD's ROCm image and then manually install our dependencies. Unfortunately, I would get cryptic error messages while running ``sudo apt-get ...``. It's still a little unclear to me whether:

- this was an image-specific issue. In other words: were the problems arising because the ROCm image was overwriting the repositories to download packages from? (seems fairly unlikely).
- this was an issue related to the docker image's permissions?
- this is a docker "feature" (e.g. for security? related to docker layers?)

In any case, I don't know enough about it to fix it.
