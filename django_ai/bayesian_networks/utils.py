# -*- coding: utf-8 -*-

import numpy as np
import pyparsing as pp
from importlib import import_module

from django.conf import settings


# Load all the modules
if hasattr(settings, "DJANGO_AI_WHITELISTED_MODULES"):
    allowed_modules = settings.DJANGO_AI_WHITELISTED_MODULES
else:
    # Default modules
    allowed_modules = [
        "numpy",
        "bayespy.nodes"
    ]
# Import all the WL modules
modules = {}
for module in allowed_modules:
    modules[module] = import_module(module)


def eval_function(parsed_fun):
    """
    Eval a function without the need of sanitizing
    """
    if parsed_fun[0] == "@":
        parsed_fun.pop(0)
        reference = True
    else:
        reference = False
    try:
        namespace, functor = parsed_fun.asList()[0].rsplit(".", 1)
    except Exception as e:
        raise ValueError("Functions must be namespaced")
    functor_args = parsed_fun[1:]
    try:
        mod = modules[namespace]
    except KeyError:
        raise ValueError("Module / Namespace not allowed, whitelist it first")
    try:
        method = getattr(mod, functor)
        if reference:
            return(method)
        else:
            # No support for kwargs yet
            return(method(*functor_args))
    except Exception as e:
        msg = e.args[0]
        raise ValueError("Invalid function invocation:" + msg)


def parse_node_args(args_string, flat=False):
    """
    Parses the string intended for Node Initialization
    Based on https://groups.google.com/forum/#!msg/comp.lang.python/vgOWCZ7Z8Yw/ZQgDJfXtCj0J
    """
    LPAR, RPAR, LBRACK, RBRACK, EQ, COMMA = \
        map(pp.Suppress, "()[]=,")

    # Booleans and Keywords
    noneLiteral = pp.Literal("None")
    boolLiteral = pp.oneOf("True False")
    # Scalars
    integer = pp.Combine(pp.Optional(pp.oneOf("+ -")) +
                         pp.Word(pp.nums)).setName("integer")
    real = pp.Combine(pp.Optional(pp.oneOf("+ -")) +
                      pp.Word(pp.nums) + pp.oneOf(". e") +
                      pp.Optional(pp.oneOf("+ -")) +
                      pp.Optional(pp.Word(pp.nums))).setName("real")
    # Identifiers
    identifier = pp.Word(pp.alphas + "_:", pp.alphanums + "_:")
    funStr = pp.Forward().setResultsName("fun")
    # Structures
    listStr = pp.Forward()
    tupleStr = pp.Forward()

    listItem = (real | integer |
                noneLiteral | boolLiteral |
                pp.quotedString.setParseAction(pp.removeQuotes) |
                pp.Group(listStr()) | tupleStr() | identifier)

    funStr << (pp.Optional("@") +
               pp.delimitedList(identifier, delim=".", combine=True) +
               (LPAR + pp.Optional(pp.delimitedList(listItem)) + RPAR))
    listStr << (LBRACK + pp.Optional(pp.delimitedList(listItem)) +
                pp.Optional(COMMA) + RBRACK)
    tupleStr << (LPAR + pp.Optional(pp.delimitedList(listItem)) +
                 pp.Optional(COMMA) + RPAR)
    kwarg = (pp.Group(identifier("kwarg") + EQ +
                      (pp.Group(funStr("fun")) | listItem()
                       ).setResultsName("kwvalue")))

    arg = pp.Group(funStr("fun")) | kwarg("kwarg") | listItem
    args = pp.delimitedList(arg)

    # parse actions perform parse-time conversions
    noneLiteral.setParseAction(lambda: None)
    boolLiteral.setParseAction(lambda toks: toks[0] == "True")
    integer    .setParseAction(lambda toks: int(toks[0]))
    real       .setParseAction(lambda toks: float(toks[0]))
    listStr    .setParseAction(lambda toks: toks.asList())
    tupleStr   .setParseAction(lambda toks: tuple(toks.asList()))
    funStr     .setParseAction(lambda toks: toks.asList())

    parsedArgs = args.parseString(args_string)

    f_args = []
    f_kwargs = {}

    for item in parsedArgs:
        if isinstance(item, pp.ParseResults):
            if 'kwarg' in item:
                if 'fun' in item.kwvalue:
                    kw_value = eval_function(item.kwvalue)
                    f_kwargs[item.kwarg] = kw_value
                else:
                    kw = item.asList()[0]
                    kw_val = item.asList()[1]
                    f_kwargs[kw] = kw_val
            else:
                if 'fun' in item:
                    f_args.append(eval_function(item))
                else:
                    f_args.append(item.asList())
        else:
            # It is one of the "basic objects" and passed as an arg
            f_args.append(item)

    if flat:
        return f_args + list(f_kwargs.values())
    else:
        return({"args": f_args, "kwargs": f_kwargs})


def mahalanobis_distance(x, y, S):
    """
    Returns the Mahalanobis distance between x and y, given the covariance
    matrix S.
    """
    S_inv = np.linalg.inv(S)
    dxy = np.array(x) - np.array(y)
    dist = np.sqrt(np.dot(np.dot(dxy.T, S_inv), dxy))
    return(dist)
