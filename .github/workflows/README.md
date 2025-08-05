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

Importantly, all of this logic is executed within a custom docker image. This docker image contains all of the dependencies that we need for building Cholla.

Historically, we would pull the custom docker image from DockerHub, but (at the time of writing) we're in the process of transitioning to using the GitHub Container Registry. We provide more context about building down below.

## Building Images

Periodically, we need to bump the versions of compilers/libraries used within the docker image that we use for the compilation check. To do this, it's important to understand how an image gets built. Consequently,
- This section provides a basic overview of the details of building new images. The purpose is to provide the reader with a basic understanding of the various concepts.
- The next section outlines the actual procedure that needs to be followed in order to update the image used by the Compilation-Check.


> [!IMPORTANT]  
> We will provide a procedure in the next section for updating the Docker Image used for the Compilation Checks. Be advised, simply updating a Dockerfile is **NOT** enough to do that.

### I. The recipe of image: **Dockerfile**

The contents of an image are dictated by the **Dockerfile**s found within the **docker/** directory. You should think of this as the recipe for the image (i.e. they provide the instructions to create an image).

### II. Building the Recipe
This is where the workflow in **build_image.yml** becomes relevant. The workflow is used to build the image from the Dockerfile

In more detail, this workflow is automatically triggered in PRs where we update this or we update one of the **Dockerfile**s.
- when the workflow runs, it builds a new Docker image from the ROCm Dockerfile
- when you manually trigger the workflow, the new docker image is uploaded to the GitHub container registry (ghcr.io)
  - **IMPORTANTLY:** a new image will **NOT** be uploaded in when the workflow is triggered by a Pull Request (to avoid the creation of many, unnecessary images)
  - **How to manually trigger:** this is a simple procedure (be aware, GitHub might change how exactly you do this in the future)
    1. Navigate to the landing page for the Cholla repository
    2. Click on the "Actions" tab
    3. On the left sidebar, you will see a list of actions. Click on the ``Build-Image`` action.
    4. Now, in the table at the center of the page, the top row should state that "This workflow has a `workflow_dispatch` event trigger." All the way on the right side of this column, there is a drop-down button that says "Run workflow". You should run the workflow from the desired branch (usually the dev branch).


## How do I update the image that we use for Compilation Checks?

This is a multi-step procedure for doing this.

1. open a new PR (to the dev branch), where you update the ROCm **Dockerfile**. The Dockerfile is the **only** thing that should change in the PR.

2. Somebody needs to review and merge the PR.
   - When the PR is made, the ``Build-Image`` will automatically run to try to build the image (but it won't upload the resulting image).
   - If the workflow fails to build an image, that's almost certainly that the **Dockerfile** has an error.

3. Once the Dockerfile-Update PR has been merged, it's now time to update to manually trigger the ``Build-Image`` workflow. As we previously noted up above, when the ``Build-Image`` workflow is manually triggered, it will upload the image to the GitHub Container Registry. (The procedure for manually triggering the workflow is described up above).

4. Wait a few minutes after step 3 is done to confirm that the image was successfully uploaded.
   - You can see a list of all images built in this way by clicking on the packages button on the right sidebar of the main GitHub webpage for the Cholla repository.
   - At the time of writing, each version of the image is named based on the Git Commit that the image was built from.

5. To actually use the new image in the Compilation Checks workflow, you need to modify the path of the container listed in `jobs.Build.strategy.matrix.container.link`. You should do this in a separate PR.

## Other Thoughts

As an experiment, I tried to see if we could get away with running the Compilation Tests without constructing a custom image. In more detail, I tried to make the compilation tests run using AMD's ROCm image and then manually install our dependencies. Unfortunately, I would get cryptic error messages while running ``sudo apt-get ...``. It's still a little unclear to me whether:

- this was an image-specific issue. In other words: were the problems arising because the ROCm image was overwriting the repositories to download packages from? (seems fairly unlikely).
- this was an issue related to the docker image's permissions?
- this is a docker "feature" (e.g. for security? related to docker layers?)

In any case, I don't know enough about it to fix it.
