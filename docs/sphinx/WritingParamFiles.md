# Writing Parameter Files

At the time of writing, Cholla's runtime files are written using a custom subset of the [TOML-format](https://toml.io/en/).
A number of examples can be found in the {repository-file}`examples` directory.

At a high-level, Cholla's parameter files are consist of the following 3 components:

- comments
- key/value pairs
- tables

We describe these components in more detail down below.

:::{note}
The goal is fully embrace the TOML format in the near future
:::

## Comment

A comment is any line where the first character on the line is a number sign, `#`.

```toml
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

```toml
# this line is a comment. The next line is not a comment.
my_parameter=3
```

At this time, a value can be either a string, boolean, integer, or float.

### Keys

Currently, we require that all keys satisfy the requirements of TOML's "bare keys."
In other words a key is composed of one or more of the following characters: `A-Za-z0-9_-`

### String

Currently, we require strings to be composed of ASCII characters.

The following snippet illustrates the 2 supported approaches for specifying a string.

```toml
indir="path/to/input-data"
outdir='path/to/output'
```

In the snipet, the value associated with `indir` is a "basic string" (i.e. it is enclosed by a pair of `"` characters), while the value associated with `outdir` is a "raw string" (it is enclosed by a pair of `'` characters.
A "basic string" supports escaped characters (for now, we forbid escaped characters of the form `\uXXXX` or `\UXXXXXXXX`).
Precise definitions are provided [here](https://toml.io/en/v1.0.0#string).

:::{note}
At this time, TOML's multiline strings charaters are expressly forbidden.
:::

### Boolean

In the following snippet, the parameters are set to the 2 boolean values.

```toml
my_param_1=true
my_param_2=false
```

### Integers and Floats

At the moment, we use [std::strtoll](https://en.cppreference.com/w/cpp/string/byte/strtol) and [std::strtod](https://en.cppreference.com/w/cpp/string/byte/strtof.html) to parse integers and floating point values.

:::{note}
Be aware that TOML's formal requirements for [integers](https://toml.io/en/v1.0.0#integer) and [floats](https://toml.io/en/v1.0.0#float) may slightly differ from the behavior of the linked functions. We encourage users to write these values down as if you were writing a literal in C/C++ to avoid these edge cases.
:::

## Tables

Key-value pairs can be organized into tables.
The name of a table is always enclosed in square-brackets.

The most instructive way to describe how tables work is to provide a concrete example.
Suppose we had the following file:

```toml
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

