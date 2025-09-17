# Writing and Building Docs

The Cholla documentation is written in a plaintext markup language, called MyST Markdown, and converted into a website with the [Sphinx](https://www.sphinx-doc.org/) documentation generator.

## What is MyST Markdown

There are *MANY* flavors of Markdown with various extensions.
[MyST](https://myst-parser.readthedocs.io/en/latest/index.html) ("Markedly Structured Text") extends the CommonMark Markdown specification to address the deficiencies in the core Markdown language that makes it a subpar choice for writing extensive and modern documentation.[^poor-markdown-docs].

The [myst-parser](https://myst-parser.readthedocs.io/en/latest/index.html) is intended to integrate with the [Sphinx](https://www.sphinx-doc.org/) documentation generator.
Sphinx was originally designed to generate documentation from the ReStructuredText markup language,[^rst] which implements extensions in terms of roles and directives.
Consequently, the MyST Markdown focuses on implements syntax extensions for implementing roles and directives so that it achieves parity with RestructuredText.

We have chosen to adopt MyST markdown, rather than ReStructuredText, because (i) people are much more likely to be familiar with Markdown and (ii) it will be easier to transition our existing documentation.
While it will probably be marginally easier for a first time Cholla contributor to write "normal" text, there is a tradeoff.
Specifically, there is probably a steeper learning curve for using Sphinx's directives and roles since most examples are written in terms of ReStructuredText.
In other words, you may need to "translate" an example to an equivalent snippet that is written in terms MyST.
Having said that, I suspect that the situation will improve over time since MyST-with-Sphinx seems to be rapidly growing in popularity (mainstream projects like pip have documentation written in MyST).

## Building a local copy of the documentation

To locally build the documentation, you must:

1. Manually install a few (non-python) dependencies:
   - [Doxygen](https://www.doxygen.nl/): it is invoked as a subprocess so that the output can be used with the {ref}`breathe sphinx extension <sphinx-extension/breathe>`
   - [pandoc](https://pandoc.org/): it is invoked by the [nbsphinx](https://nbsphinx.readthedocs.io/en/0.9.7/) sphinx extension to help generate webpages from jupyter notebooks.

2. Then you should invoke the following commands from the root of the Cholla repository.
   (We suggest that you do this in a virtual python environment).

   ```shell-session
   # ensure that pip is up to date (we need version 25.1+)
   $ python -m pip install --upgrade pip
   # install the python-dependencies
   $ python -m pip install --group dev
   # actually build the documentation
   $ python -m sphinx -M html docs/sphinx "_build" -W
   ```

   The final command puts the generated documentation in a file called **_build**.
   Be aware that the `-W` flag tells sphinx to treat warnings as errors (if you are debugging, it is sometimes useful to omit this flag).

At this point, you can render the documentation.
If FireFox is installed, this can be accomplished by invoking

::::{tab} Linux
```shell-session
$ firefox _build/html/index.html
```
::::

::::{tab} macOS
:::{tab} firefox
```shell-session
$ open -a firefox _build/html/index.html
```
:::
:::{tab} chrome
```
$ open -a "Google Chrome" _build/html/index.html
```
:::
::::

## Extensions

(sphinx-extension/breathe)=

### The `breathe` extension

To reduce some duplication, we make use of [Breathe](https://www.breathe-doc.org/).
It is a Sphinx plugin that we use to selectively integrate documentation generated with [Doxygen](https://www.doxygen.nl/) from docstrings in Cholla's source code.

:::{warning}
At the time of writing, we had to temporarily stop rendering extracting docstrings with Breathe, but doxygen is still run.
There are a bunch of errors right now (I suspect this happened because doxygen is not consistently run in the contintuous integration)
:::

### The `sphinx-inline-tabs` extension

This is a [simple extension](https://sphinx-inline-tabs.readthedocs.io/en/latest/) that make it easy to inject tabs

### The `par` extension

We have also written adopted and modified an extension for Sphinx to assist with formatting parameters, called `par`. This extension was originally written so that it could be used with Cello/Enzo-E

At present, this extension provides 3 primary features:

```{eval-rst}
1. A directive called :rst:dir:`par:parameter` that is to be used when defining a new parameter in the reference section.
   Parameters defined with this directive automatically provide an anchor point to facillitate cross-referencing (i.e. using the :rst:role:`par:param` role).

2. Pretty-formatting of parameter names.
   This is done by the :rst:dir:`par:parameter` directive, as well as the :rst:role:`par:param` and :rst:role:`par:paramfmt` roles.
   
   .. COMMENT: Note, this functionallity supports special formatting of components of parameter names enclosed by angle brackets (e.g. ``{par:paramfmt}`Boundary:<condition>:type``` and ``{par:paramfmt}`Initial:value:<field>``` are respectively rendered as :par:paramfmt:`Boundary:<condition>:type` and :par:paramfmt:`Initial:value:<field>`)

3. Pretty-formatting of parameter types with the :rst:role:`par:typefmt` role.
   For example, ``{par:typefmt}`str``` gets rendered as :par:typefmt:`str`.

```


#### Defining a Parameter

When defining a new parameter in the reference section, you should use the following directive:

:::::{rst:directive} par:parameter

This directive should be used when defining a new runtime parameter.

The argument passed to the directive is the name of the parameter being documented.
After the line `:::{par:parameter} ParamName`, you should leave a blank line, and then you should define a list of fields (e.g. ``Summary``, ``Type``, ``Default``) that provide an overview of the parameter.
Finally, you generally provide an extended description after the field-list.

**Example**

It is instructive to consider a concrete example.
Below we show how to do this for the {par:paramfmt}`chemistry.data_file` parameter:

```{code-block} md

:::{par:parameter} chemistry.data_file

:Summary: path to data file used by "tabulated-cooling" solver
:Type:    {par:typefmt}`str`
:Default: *None*

It is an error to specify this parameter when {par:param}`chemistry.kind` chemistry cooling solver other than "tabulated-cooling."
:::
```

This snippet will be rendered as the following:[^disable-cross-ref]

% As the footnote mentions, we pass the ``:noindex:`` option to the directive in order
% to disable cross-referencing.

>  :::{par:parameter} chemistry.data_file
>  :noindex:
>
>  :Summary: path to data file used by "tabulated-cooling" solver
>  :Type:    {par:typefmt}`str`
>  :Default: *None*
>
>  It is an error to specify this parameter when {par:param}`chemistry.kind` chemistry cooling solver other than "tabulated-cooling."
>  :::

**Other potential benefits of** {rst:dir}`par:parameter`

Defining parameters with this directive could also facillitate other benefits in the future such as:

* automated populating of parameter tables displayed in physics-module documentation
* the automated formatting of the text in the parameter fields
* the displayed format can be easily updated
:::::

#### Formatting parameters names in text

The extension also defines two roles that can be used when mentioning parameters in text.

::::{rst:role} par:param

This role formats the name of a parameter nicely and creates a cross reference to an existing parameter definition.


```{eval-rst}
For example, ``{par:param}`chemistry.data_file``` renders as :par:param:`chemistry.data_file`.

You can modify this role's behavior by prefixing the content with a ``~`` or ``!``:

* when the role's content is prefixed with a ``~``, the rendered link text only shows the final component of the parameter.
  For example, ``{par:param}`~chemistry.data_file``` renders as :par:param:`~chemistry.data_file` (it links to the same place as :par:param:`chemistry.data_file`).

* when the role's content is prefixed with a ``!``, no link is constructed. In other words, writing something like ``{par:param}`!chemistry.data_file``` renders as :par:param:`!chemistry.data_file`).
  This does the same thing as :rst:role:`par:paramfmt`
```

:::{note}
Parameter names that also serve as links are intentionally styled different from non-linked parameter names (to provide readers with a visual cue to the difference).
:::
::::

::::{rst:role} par:paramfmt

This role is provided as a convenience.
It effectively aliases the behavior of the {rst:role}`par:param` role when the content is prefixed by ``!`` (i.e. the name is formatted nicely, but no link is created).

```{eval-rst}
For example, ``{par:paramfmt}`chemistry.data_file``` renders as :par:paramfmt:`chemistry.data_file`.
```

::::

#### Formatting parameter types in text

We have defined the following role to nicely format the names of parameter types.

::::{rst:role} par:typefmt

The impact of this role is best illustrated by example:

% (there doesn't seem to be a way to write out a role in a codespan in myst-syntax)
```{eval-rst}
* ``{par:typefmt}`float``` renders as :par:typefmt:`float`
* ``{par:typefmt}`str``` renders as :par:typefmt:`str`
```
::::

[^poor-markdown-docs]: To be clear, this isn't a slight against Markdown.
    It is indeed possible to write documentation in pure Markdown (but you are missing features and it is more labor intensive).
    However the deficiencies are clearly illustrated by the fact that so many different Markdown dialects introduce their own forms of extensions.
    For example the GitHub dialect and the dialect for documenting Julia code have separate custom syntaxes for creating admonitions.
    The dialects for documenting Julia and documenting Rust also have separate custom syntaxes for cross-referencing to the code symbols (e.g. function names or struct names).
    Support for LaTeX math is another common extension.
    Furthermore, MkDocs and MyST both implement syntaxes for introducing general extensions.
    Plus, some dialects encourage you to directly embed HTML into your documentation to achieve certain features (e.g. GitHub for Collapsable sections, Rust for warnings).

[^rst]: ReStructuredText was originally created for use in Python docstrings (it's about 3 years older than Markdown).
    There are a lot of similarities with Markdown.
    Unlike Markdown, ReStructuredText is always valid ReStructuredText (i.e. there's no questions of vendor-specific dialects), extensibility is built into the core language, and its easier to specify complex tables.
    For these reasons, I think it's generally a better choice for writing documentation.
    With that said, there appears to be a general trend towards adopting Markdown for writing documentation.

[^disable-cross-ref]: To avoid creating confusion with other parameters, we have artificially disabled cross-referencing.
    If you wanted to render a parameter and disable cross-referencing, you need to pass the ``:noindex:`` option to the directive to the {rst:dir}`par:parameter` parameter.

