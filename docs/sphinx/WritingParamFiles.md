# Writing Parameter Files

At the time of writing, Cholla's runtime files are written in a custom [INI-format](https://en.wikipedia.org/wiki/INI_file).
A number of examples can be found in the {repository-file}`examples` directory.

At a high-level, Cholla's parameter files are consist of the following 3 components:

- comments
- key/value pairs
- tables

We describe these components in more detail down below.

:::{note}
We have been slowly migrating towards using the visually-similar [TOML format](https://toml.io/en/).
More recent features (e.g. tables), have been added in a manner consistent with the TOML format.
:::

## Comment

A comment is any line where the first character on the line is a number sign, `#`.

```ini
# this line is a comment. The next line is not a comment.
my_parameter=3
```

:::{important}
Cholla currently encounters issues if you try to write a comment when the first character is on a line isn't a number sign
:::

## Key/Value Pairs

Information is primarily specified in Parameter Files through the use of Key/Value pairs.
A Key/Value pair has the form `<key>=<value>`, where `<key>` is replaced with the name of a parameter and `<value>` is replaced with the value that you want to associate with the parameter.

:::{important}
Currently, Cholla does **NOT** properly handle whitespace around the equal sign character character
:::

We illustrate a concrete example down below:

```ini
# this line is a comment. The next line is not a comment.
my_parameter=3
```

## Tables

Key-value pairs can be organized into tables.
The name of a table is always enclosed in square-brackets.

The most instructive way to describe how tables work is to provide a concrete example.
Suppose we had the following file:

```ini
my_param=3442

[my_section_a]
my_param=-1

[my_section_a.subsection]
my_param=2

[my_section_b]
my_param=14
```

In this file we have 4 distinct parameters:

1. `my_param` (a parameter in the implicit global table)
2. `my_section_a.my_param` (the parameter is part of the `my_section_a` table)
3. `my_section_a.subsection.my_param` (the parameter is part of the `my_section_a.subsection` table)
4. `my_section_b.my_param` (the parameter is part of the `my_section_b` table)

Note how in the above list, we use the dot notation to delimit parts of a full parameter name.
We use this convention throughout the codebase and the documentation.

:::{important}
A parameter file is invalid (even if Cholla does not outright reject it) to have a parameter and table with the same name.
:::

