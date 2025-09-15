"""par.py sphinx extension

This module is used to implement the par.py sphinx extension. This extension
is designed to implement the ``par`` domain for documenting and
cross-referencing Cholla's runtime parameters.


Sphinx extension that introduces the par domain for documenting and 
cross-referencing parameters

To customize formatting when generating HTML output, define CSS entries for
    .param-style
    .param-style-var-comp
    .param-type-style

Notes
-----
This module has been adapted from an extension that was originally created for
documenting Enzo-E's parameters
"""
import re
from typing import Any, Dict, List, Iterable, Sequence, Set, Tuple, TYPE_CHECKING

from docutils import nodes
from docutils.parsers.rst.directives import unchanged_required, flag

from sphinx.domains import Domain
from sphinx.roles import XRefRole
from sphinx.util.docutils import SphinxDirective, SphinxRole
from sphinx.util.nodes import make_id, make_refnode, nested_parse_with_titles

if TYPE_CHECKING:
    try:
        from sphinx.util.typing import OptionSpec
    except ImportError:
        OptionSpec = Any
    from sphinx.addnodes import pending_xref
    from sphinx.application import Sphinx
    from sphinx.builder import Builder
    from sphinx.config import _ConfigRebuild
    from sphinx.config import Config as SphinxConfig
    from sphinx.environment import BuildEnvironment
    from sphinx.util.typing import ExtensionMetadata

else:
    OptionSpec = Any
    pending_xref = Any
    Sphinx = Any
    Builder = Any
    _ConfigRebuild = Any
    SphinxConfig = Any
    BuildEnvironment = Any
    ExtensionMetadata = Any

# the following object is inspired by the napoleon extension
class Config:
    """par extension settings (configured in `conf.py`)

    This primarily exists as a way to document the extra attributes that
    appended on the global sphinx Config object

    Attributes
    ----------
    par_separator
        Specifies the separator between table names and parameter names
    par_never_spaced_separator
        A boolean that indicates whether we ever want to permit a set of
        components to be formatted as `{comp0} {sep} {comp1}` or if it
        always must be formatted as `{comp0}{sep}{comp1}`.
    """
    _config_values: Sequence[tuple[str, Any, _ConfigRebuild, Set[type]]] = (
        ("par_separator", ":", "env", frozenset({str})),
        ("par_never_spaced_separator", True, "env", frozenset({bool})),
    )

    def __init__(self, **kwargs: Any):
        # initialize the default values
        n_loaded = 0
        for name, default, _, _ in self._config_values:
            value = default
            if name in kwargs:
                n_loaded += 1
                value = kwargs[name]
            setattr(self, name, default)

        # error handling
        if n_loaded != len(kwargs):
            for key in kwargs:
                if all(key != pack[0] for pack in self._config_values):
                    raise RuntimeError(
                        "key is not known to the par extension"
                    )



class parameter_node(nodes.container):
    """
    Acts as a way to group a collection of children nodes together and acts as
    an anchor point for cross-references (in principle, it might be more clean
    to split these responsibilities).

    Note
    ----
    I'm not totally sure we should actually be subclassing ``nodes.container``:

      * we are using it because it seems to group together elements in a
        non-invasive way (that doesn't affect styling)
      * we need to have some kind of concrete way to group together the
        parameters that can be used as a concrete reference location
      * if we decide not to use ``nodes.container``, we should instead subclass
        ``nodes.Element`` instead
    """
    pass

def visit_parameter_node(self, node: parameter_node):
    # self is a Writer object (like sphinx.writers.html.HTML5Translator)
    #
    # in the future, we could make special variants of this for different
    # writers for customizing formatting (e.g. add a background color)
    self.visit_container(node)

def depart_parameter_node(self, node: parameter_node):
    # self is a Writer object (like sphinx.writers.html.HTML5Translator)
    #
    # in the future, we could make special variants of this for different
    # writers for customizing formatting (e.g. add a background color)
    self.depart_container(node)

def _create_parameter_name_nodes(
    name, config: SphinxConfig, add_space_to_separator: bool = False
) -> List[nodes.Node]:
    """
    Parses a string holding a parameter name and returns a list of
    docutils nodes that can be used for styling.

    A full parameter name is composed of multiple components. Let's consider
    an example where the separator is a colon. In this case, the parameter
    "Method:heat:alpha" has components named "Method", "heat", and "alpha".

    When a part of a parameter name is enclosed by angle brackets, we call that
    a "variable-component". It's understood to be a placeholder for a different
    concrete component. As such the styling is different.

    Notes
    -----
    To support styling we put the names of components into instances of
    ``nodes.inline`` and append either 'param-style' or 'param-style-var-comp'
    to the classes attribute list.

    When producing html output outputs, the names are effectively enclosed by
    span tags, and the styling is controlled by introducing definitions css
    definitions for .param-style and .param-style-var-comp
    """

    name = name.strip() # remove leading/trailing whitespace (may not be
                        # necessary)
    separator = config.par_separator

    if name == '':
        raise ValueError("this function was just passed whitespace")
    elif name[-1] == separator:
        raise ValueError(f"the final character in '{name}' parameter's name "
                         f"is a '{separator}'")
    elif any(e.isspace() for e in name):
        # ensure that the name doesn't contain whitespace
        raise ValueError(f"'{name}' contains whitespace")

    if add_space_to_separator and not config.par_never_spaced_separator:
        make_seperator_node = lambda : nodes.Text(f' {separator} ')
    else:
        make_seperator_node = lambda : nodes.Text(separator)

    # split into parts and construct a list of nodes
    node_l = []
    for cur_part in name.split(separator):
        if len(node_l) > 0:
            node_l.append(make_seperator_node())

        if ((len(cur_part) > 2) and
            ('<' == cur_part[0]) and ('>' == cur_part[-1])):
            cur_classes_attr, test_slc = 'param-style-var-comp', slice(1,-1)
        else:
            cur_classes_attr, test_slc = 'param-style', slice(None)

        if any((e in '><') for e in cur_part[test_slc]):
            raise ValueError(f"Errant '>' or '<' found in {name}")

        cur_node = nodes.inline('', cur_part)
        cur_node['classes'].append(cur_classes_attr)
        node_l.append(cur_node)

    return node_l


class ParameterDirective(SphinxDirective):
    """
    Directive to be used for formatting a Cello/Enzo-E parameter

    Note
    ----
      * The implementation of this Directive would be much more straightforward
        if it simply subclassed the Sphinx's ``ObjectDescription`` class
      * The tradeoff from doing that is that we give up a lot of control over
        formatting - each parameter definition would resemble a function
        signature.
      * This may ultimately be a preferable choice. However, when initially
        presenting this extension, I wanted to keep the output formatting
        consistent.

    To understand what this class is doing, it can be useful to refer to:
        https://docutils.sourceforge.io/docs/howto/rst-directives.html
    (the documentation for the ``Directive`` class is particularly insightful)
    """

    has_content = True
    optional_arguments = 0
    required_arguments = 1 # the required argument is the name of the parameter
    option_spec: OptionSpec = {'noindex': flag}

    def run(self):
        # the single positional argument passed to this directive is the full
        # name of the parameter that is documented by this directive's contents
        full_name = self.arguments[0]

        noindex = ('noindex' in self.options)

        # STAGE 1: make a parameter node
        # ------------------------------
        # The parameter node will be the parent of all other nodes parsed as
        # part of this directive
        # NOTE: it's not actually essential for a directive to use a custom
        #       node type. This function can instead return a list of existing
        #       node types, but I find this conceptually useful (& it's useful
        #       for making the parameters cross-referenceable)
        my_parameter_node = parameter_node()

        # STAGE 2: make the parameter node referenceable
        # ----------------------------------------------
        #   I don't totally understand how this next bit of code works. I
        #   cobbled it together after looking at the tutorials & some
        #   directives within Sphinx. But I'll try to explain what happens

        if not noindex:
            # come up with a node_id. I think this needs to be unique (but there
            # might be some flexibility)
            # - I believe that the make_id just modifies some characters in the
            #   passed string that may be invalid
            node_id = make_id(self.env, self.state.document, '',
                              f'param-{full_name}')
            # add the node_id to the ids attribue of the parameter_node
            my_parameter_node['ids'].append(node_id)
            # register the parameter_node as an explicit target with the
            # document
            # - this updates some map of ids internally tracked by the document
            # - this access the ids attribute of the parameter_node
            self.state.document.note_explicit_target(my_parameter_node)

            # register the name of the parameter node and the node_id with
            # ParDomain. This will helps with the creation of cross-references
            # to this node at a later time
            par_domain = self.env.get_domain('par')
            par_domain.register_parameter_obj(full_name, anchor = node_id)

        # STAGE 3: parse the contents within the directive
        # ------------------------------------------------
        # the following is identical to the more conventional command
        #    self.state.nested_parse(self.content, self.content_offset,
        #                            my_parameter_node)
        # but, it preserves nested section titles included within the directive
        nested_parse_with_titles(self.state, self.content, my_parameter_node)

        # STAGE 4: include the parameter name as a field in the documentation
        # -------------------------------------------------------------------
        # we are mutating some of the parsed contents

        if len(my_parameter_node.children) == 0:
            # not sure if we should support this case
            my_parameter_node.append(nodes.field_list())

        # the expectation is that the first node child is always a field_list
        # retrieve this node
        field_list = my_parameter_node.children[0]
        if not isinstance(field_list, nodes.field_list):
            raise RuntimeError(
                f"There is a problem with the input for the {self.name} "
                f"directive that is being used to define the `{full_name}` "
                "parameter. The first set of contents in the directive (after "
                "any options) must be a field list. The directive is also "
                "allowed to be empty."
            )

        # lets create the new field node that will be called Parameter
        new_fieldname = nodes.field_name('', "Parameter")
        new_fieldbody = nodes.field_body(
            '',
            nodes.paragraph(
                '',
                '',
                *_create_parameter_name_nodes(
                    full_name, config=self.config, add_space_to_separator = True
                ),
            )
        )
        new_field = nodes.field('', new_fieldname, new_fieldbody)

        # insert the new field at the start of field list
        field_list.insert(0, new_field)

        # the following is an attempt to facillitate automatic field formatting
        # - this actually works quite nicely, but I think it may be a little
        #   confusing (so we will disable for now)
        # - we skip the very first field since we know it's the parameter name
        #self.tag_fieldbody_text(field_list.children[1:])

        return [my_parameter_node]

    def tag_fieldbody_text(self, fieldlist_children):
        """
        This method attempts to facillitate automatic field formatting for all
        text without any explicit user-specified formatting.

        In more detail, this method iterates over the fieldbody nodes included
        in the provided children of a fieldlist node. For a given fieldbody
        node, this method replaces all of its grandchildren ``Text`` nodes with
        ``inline`` nodes that contain equivalivent text. Additionally,
        f"par-fieldval-{field}" is appended to the ``"classes"`` attribute of
        each inline node (where ``field`` is the all-lowercase version of the
        fieldname that corresponds to the grandparent fieldbody).

        In practice, when an html document is rendered, each of these ``inline``
        nodes becomes span tags with the previously mentioned classname stored
        in its class attribute. To take advantage of this for a Summary field,
        one needs to define an entry for .par-fieldval-summary in a CSS file
        """
        for field in fieldlist_children:
            cur_fieldname, cur_fieldbody = field[0], field[1]
            assert isinstance(cur_fieldname, nodes.field_name)
            assert isinstance(cur_fieldbody, nodes.field_body)

            cur_fieldname_str = str(cur_fieldname[0])

            # sanity check that fieldbody only has 1 child (a paragraph node)
            assert ( (len(cur_fieldbody.children) == 1) and
                     isinstance(cur_fieldbody[0], nodes.paragraph))
            old_paragraph = cur_fieldbody.pop()

            new_paragraph = old_paragraph.copy()
            new_paragraph.clear()

            for elem in old_paragraph.children:
                if isinstance(elem, nodes.Text):
                    elem = nodes.inline('', str(elem))
                    elem['classes'].append(
                        f"par-fieldval-{cur_fieldname_str.lower()}"
                    )
                new_paragraph.append(elem)
            cur_fieldbody.append(new_paragraph)

class ParamFmtRole(SphinxRole):
    """
    Applies the same formatting as ParamXRefRole, but doesn't make links
    """

    def run(self) -> Tuple[List[nodes.Node], List[nodes.system_message]]:
        nodes = _create_parameter_name_nodes(name=self.text, config=self.config)
        return nodes, []

class paramxref_node(nodes.TextElement):
    # this is just used to give "code-like-formatting" (so that we can
    # distinguish refenerence from non-references)
    pass

def html_visit_paramxref_node(self, node: paramxref_node):
    return self.body.append('<code>')

def html_depart_paramxref_node(self, node: paramxref_node):
    return self.body.append('</code>')

def visit_paramxref_node(self, node: paramxref_node): pass
def depart_paramxref_node(self, node: paramxref_node): pass

class ParamXRefRole(XRefRole):
    """
    Used for cross-referencing.

    Extended to:
      - support abbreviated titles
      - make the formatting more consistent with ParamFmtRole
    """

    innernodeclass = nodes.inline

    def process_link(self, env: BuildEnvironment, refnode: nodes.Element,
                     has_explicit_title: bool, title: str,
                     target: str) -> Tuple[str, str]:
        # note it's possible for a cross-reference to be made with an
        # "explicit target" & explicit reference target (that don't necessarily
        # need to match). That is done with syntax like
        #   :role:`mytitle <mytarget>`
        # where role is replaced with some arbitrary syntax

        if (not has_explicit_title) and title[0] == '~':
            # leading tilde means nothing for the target (get rid of it)
            target = target[1:]
            # the leading tilde means that we just display the last component
            # of the target
            title = (title[1:].split(env.config.par_separator))[-1]

        return title, target

    def result_nodes(
        self,
        document: nodes.document,
        env: BuildEnvironment,
        node: nodes.Element,
        is_ref: bool
    ) -> Tuple[List[nodes.Node], List[nodes.system_message]]:
        # called just before the completed nodes are returned
        # -> we are going to customize formatting of the text

        if is_ref:
            # `node` refers to a reference node
            inline_node = node.children[0]
        else:
            # `node` refers directly to inline
            # this branch is only executed when cross references are disabled
            # (I think this happens when an exclamation point is prepended to
            # the cross-reference target)
            inline_node = node
        # sanity checks
        assert isinstance(inline_node, nodes.inline)
        assert len(inline_node.children) == 1

        # pop the text node out of inline_node and convert back to raw text
        original_text_node = inline_node.pop()
        raw_text = str(original_text_node)

        # build the nicely formated parameter name nodes
        param_name_nodes = _create_parameter_name_nodes(raw_text, config=self.config)
        if is_ref:
            # to visually signify that this parameter name is a link,
            # stick the nodes into a paramxref_node
            param_name_nodes = paramxref_node('', '', *param_name_nodes)

        inline_node += param_name_nodes

        #must return a (nodes,messages) tuple
        return [node], []

# the split method of the following object is used to produce a list that
# alternates between non-type parameters & non-type parameters
# - even entries of the resulting list get special formatting
# - odd entries don't get that formatting
_TYPE_EXPR_PATTERN = re.compile(r'([-a-zA-Z]+)')

class TypeFmtRole(SphinxRole):
    """
    Formats a type of a parameter.

    In practice, this encloses all ascii-letter characters and the hyphen
    character in instances of `nodes.inline` (with "param-type-style" appended
    to the classes attribute).
    """

    def run(self) -> Tuple[List[nodes.Node], List[nodes.system_message]]:
        name = self.text.strip()

        node_l = []
        for i, elem in enumerate(_TYPE_EXPR_PATTERN.split(name)):

            # when ``i`` is odd then we place elem in a node that specifies
            # the special formatting.
            if (i % 2) == 1:
                node_l.append(nodes.inline('', elem))
                node_l[-1]['classes'].append('param-type-style')
            elif elem == '':
                # this is effectively a placeholder that arises when i == 0 &
                # the very first character belongs to the group of characters
                # that gets special formatting (this is effectively a
                # placeholder to ensure that rules about even and odd values of
                # i are still applicable)
                continue
            else:
                node_l.append(nodes.Text(elem))
        return node_l, []

class ParDomain(Domain):
    name = 'par'
    label = 'par'

    roles = {
        'param': ParamXRefRole(),
        'paramfmt' : ParamFmtRole(),
        # reserve the 'type' role for a potential future where we want to
        # support linking to a description of the type
        'typefmt' : TypeFmtRole()
    }
    directives = {
        "parameter" : ParameterDirective
    }

    initial_data: Dict[str, Dict[str, Tuple[str, str]]] = {
        'parameters': {},  # each key is the fully qualified name (& dispname)
                           # each associated value is a tuple of: (
                           #   docname: document name where target is found
                           #   node_id: aka the anchor name
    }

    @property
    def parameters(self) -> Dict[str, Tuple[str, str]]:
        return self.data.setdefault('parameters', {})

    def register_parameter_obj(self, full_name:str, anchor:str):
        """
        Adds a new parameter object to the domain

        This is called by ParameterDirective
        """
        self.parameters[full_name] = (self.env.docname, anchor)

    def get_objects(self) -> Iterable[Tuple[str, str, str, str, str, int]]:
        for full_name, (docname, anchor) in self.parameters.items():
            dispname = full_name
            # TODO: if we add more object types in the future, can't hardcode
            #       type_string
            type_str = 'Parameter'
            yield (full_name, dispname, type_str, docname, anchor, 1)

    def resolve_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        typ: str,
        target: str,
        node: pending_xref,
        contnode: nodes.Element
    ) -> nodes.reference | None:
        # if we add more object types, we may want to refactor
        try:
            todocname, anchor = self.parameters[target]
            return make_refnode(builder, fromdocname, todocname, anchor,
                                contnode, anchor)
        except KeyError:
            print('could not find reference')
            return None

def setup(app: Sphinx) -> ExtensionMetadata:
    """Set up the par extension

    Parameters
    ----------
    app
        App object that represents the underlying Sphinx process
    """

    app.add_node(parameter_node,
                 html=(visit_parameter_node, depart_parameter_node),
                 latex=(visit_parameter_node, depart_parameter_node),
                 text=(visit_parameter_node, depart_parameter_node),
                 man=(visit_parameter_node, depart_parameter_node),
                 texinfo=(visit_parameter_node, depart_parameter_node)
    )
    app.add_node(paramxref_node,
                 html=(html_visit_paramxref_node, html_depart_paramxref_node),
                 latex=(visit_paramxref_node, depart_paramxref_node),
                 text=(visit_paramxref_node, depart_paramxref_node),
                 man=(visit_paramxref_node, depart_paramxref_node),
                 texinfo=(visit_paramxref_node, depart_paramxref_node),
    )
    app.add_domain(ParDomain)

    for name, default, rebuild, types in Config._config_values:
        app.add_config_value(name, default, rebuild, types=types)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
    }

