#!/usr/bin/env python3
# Copyright 2017 The Dawn Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, os, sys
from collections import namedtuple

# sys.path.insert(1, 'third_party/dawn/generator')
kDawnGeneratorPath = '--dawn-generator-path'
try:
    dawn_generator_path_argv_index = sys.argv.index(kDawnGeneratorPath)
    path = sys.argv[dawn_generator_path_argv_index + 1]
    sys.path.insert(1, path)
except ValueError:
    # --dawn-generator-path isn't passed, ignore the exception and just import
    # assuming it already is in the Python PATH.
    print('No dawn generator path defined')
    sys.exit(1)

from generator_lib import Generator, run_generator, FileRender
from dawn_json_generator import parse_json, Method, Name

#############################################################
# Generator
#############################################################


def as_varName(*names):
    return names[0].camelCase() + ''.join(
        [name.CamelCase() for name in names[1:]])


def as_cType(name):
    if name.native:
        return name.concatcase()
    else:
        return 'Webnn' + name.CamelCase()


def as_cTypeDawn(name):
    if name.native:
        return name.concatcase()
    else:
        return 'Dawn' + name.CamelCase()


def as_cTypeEnumSpecialCase(typ):
    if typ.category == 'bitmask':
        return as_cType(typ.name) + 'Flags'
    return as_cType(typ.name)


def as_cppType(name):
    if name.native:
        return name.concatcase()
    else:
        return name.CamelCase()


def as_jsEnumValue(value):
    if value.jsrepr: return value.jsrepr
    return "'" + value.name.js_enum_case() + "'"


def convert_cType_to_cppType(typ, annotation, arg, indent=0):
    if typ.category == 'native':
        return arg
    if annotation == 'value':
        if typ.category == 'object':
            return '{}::Acquire({})'.format(as_cppType(typ.name), arg)
        elif typ.category == 'structure':
            converted_members = [
                convert_cType_to_cppType(
                    member.type, member.annotation,
                    '{}.{}'.format(arg, as_varName(member.name)), indent + 1)
                for member in typ.members
            ]

            converted_members = [(' ' * 4) + m for m in converted_members]
            converted_members = ',\n'.join(converted_members)

            return as_cppType(typ.name) + ' {\n' + converted_members + '\n}'
        else:
            return 'static_cast<{}>({})'.format(as_cppType(typ.name), arg)
    else:
        return 'reinterpret_cast<{} {}>({})'.format(as_cppType(typ.name),
                                                    annotation, arg)


def decorate(name, typ, arg):
    if arg.annotation == 'value':
        return typ + ' ' + name
    elif arg.annotation == '*':
        return typ + ' * ' + name
    elif arg.annotation == 'const*':
        return typ + ' const * ' + name
    elif arg.annotation == 'const*const*':
        return 'const ' + typ + '* const * ' + name
    else:
        assert False


def annotated(typ, arg):
    name = as_varName(arg.name)
    return decorate(name, typ, arg)


def as_cEnum(type_name, value_name):
    assert not type_name.native and not value_name.native
    return 'Webnn' + type_name.CamelCase() + '_' + value_name.CamelCase()


def as_cEnumDawn(type_name, value_name):
    assert not type_name.native and not value_name.native
    return ('DAWN' + '_' + type_name.SNAKE_CASE() + '_' +
            value_name.SNAKE_CASE())


def as_cppEnum(value_name):
    assert not value_name.native
    if value_name.concatcase()[0].isdigit():
        return "e" + value_name.CamelCase()
    return value_name.CamelCase()


def as_cMethod(type_name, method_name):
    assert not type_name.native and not method_name.native
    return 'webnn' + type_name.CamelCase() + method_name.CamelCase()


def as_cMethodDawn(type_name, method_name):
    assert not type_name.native and not method_name.native
    return 'dawn' + type_name.CamelCase() + method_name.CamelCase()


def as_MethodSuffix(type_name, method_name):
    assert not type_name.native and not method_name.native
    return type_name.CamelCase() + method_name.CamelCase()


def as_cProc(type_name, method_name):
    assert not type_name.native and not method_name.native
    return 'Webnn' + 'Proc' + type_name.CamelCase() + method_name.CamelCase()


def as_cProcDawn(type_name, method_name):
    assert not type_name.native and not method_name.native
    return 'Dawn' + 'Proc' + type_name.CamelCase() + method_name.CamelCase()


def as_frontendType(typ):
    if typ.category == 'object':
        return typ.name.CamelCase() + 'Base*'
    elif typ.category in ['bitmask', 'enum']:
        return 'webnn::' + typ.name.CamelCase()
    elif typ.category == 'structure':
        return as_cppType(typ.name)
    else:
        return as_cType(typ.name)


def c_methods(types, typ):
    return typ.methods + [
        Method(Name('reference'), types['void'], []),
        Method(Name('release'), types['void'], []),
    ]


def get_c_methods_sorted_by_name(api_params):
    unsorted = [(as_MethodSuffix(typ.name, method.name), typ, method) \
            for typ in api_params['by_category']['object'] \
            for method in c_methods(api_params['types'], typ) ]
    return [(typ, method) for (_, typ, method) in sorted(unsorted)]


def has_callback_arguments(method):
    return any(arg.type.category == 'callback' for arg in method.arguments)


class MultiGeneratorFromWebnnJSON(Generator):
    def get_description(self):
        return 'Generates code for various target from Dawn.json.'

    def add_commandline_arguments(self, parser):
        allowed_targets = [
            'webnn_headers', 'webnncpp_headers', 'webnncpp', 'webnn_proc',
            'mock_webnn', 'webnn_native_utils'
        ]

        parser.add_argument('--webnn-json',
                            required=True,
                            type=str,
                            help='The WebNN JSON definition to use.')
        parser.add_argument(
            '--targets',
            required=True,
            type=str,
            help=
            'Comma-separated subset of targets to output. Available targets: '
            + ', '.join(allowed_targets))
        parser.add_argument(
            '--dawn-generator-path',
            required=True,
            type=str,
            help='The path of Dawn generator')

    def get_file_renders(self, args):
        with open(args.webnn_json) as f:
            loaded_json = json.loads(f.read())
        api_params = parse_json(loaded_json)

        targets = args.targets.split(',')

        base_params = {
            'Name': lambda name: Name(name),
            'as_annotated_cType': \
                lambda arg: annotated(as_cTypeEnumSpecialCase(arg.type), arg),
            'as_annotated_cppType': \
                lambda arg: annotated(as_cppType(arg.type.name), arg),
            'as_cEnum': as_cEnum,
            'as_cEnumDawn': as_cEnumDawn,
            'as_cppEnum': as_cppEnum,
            'as_cMethod': as_cMethod,
            'as_cMethodDawn': as_cMethodDawn,
            'as_MethodSuffix': as_MethodSuffix,
            'as_cProc': as_cProc,
            'as_cProcDawn': as_cProcDawn,
            'as_cType': as_cType,
            'as_cTypeDawn': as_cTypeDawn,
            'as_cppType': as_cppType,
            'as_jsEnumValue': as_jsEnumValue,
            'convert_cType_to_cppType': convert_cType_to_cppType,
            'as_varName': as_varName,
            'decorate': decorate,
            'c_methods': lambda typ: c_methods(api_params['types'], typ),
            'c_methods_sorted_by_name': \
                get_c_methods_sorted_by_name(api_params),
        }

        renders = []

        if 'webnn_headers' in targets:
            renders.append(
                FileRender('webnn.h', 'src/include/webnn/webnn.h',
                           [base_params, api_params]))
            renders.append(
                FileRender('webnn_proc_table.h',
                           'src/include/webnn/webnn_proc_table.h',
                           [base_params, api_params]))

        if 'webnncpp_headers' in targets:
            renders.append(
                FileRender('webnn_cpp.h', 'src/include/webnn/webnn_cpp.h',
                           [base_params, api_params]))

        if 'webnn_proc' in targets:
            renders.append(
                FileRender('webnn_proc.c', 'src/webnn/webnn_proc.c',
                           [base_params, api_params]))

        if 'webnncpp' in targets:
            renders.append(
                FileRender('webnn_cpp.cpp', 'src/webnn/webnn_cpp.cpp',
                           [base_params, api_params]))

        if 'emscripten_bits' in targets:
            renders.append(
                FileRender('webnn_struct_info.json',
                           'src/webnn/webnn_struct_info.json',
                           [base_params, api_params]))
            renders.append(
                FileRender('library_webnn_enum_tables.js',
                           'src/webnn/library_webnn_enum_tables.js',
                           [base_params, api_params]))

        if 'mock_webnn' in targets:
            mock_params = [
                base_params, api_params, {
                    'has_callback_arguments': has_callback_arguments
                }
            ]
            renders.append(
                FileRender('mock_webnn.h', 'src/webnn/mock_webnn.h',
                           mock_params))
            renders.append(
                FileRender('mock_webnn.cpp', 'src/webnn/mock_webnn.cpp',
                           mock_params))

        if 'webnn_native_utils' in targets:
            frontend_params = [
                base_params,
                api_params,
                {
                    # TODO: as_frontendType and co. take a Type, not a Name :(
                    'as_frontendType': lambda typ: as_frontendType(typ),
                    'as_annotated_frontendType': \
                        lambda arg: annotated(as_frontendType(arg.type), arg),
                }
            ]

            renders.append(
                FileRender('webnn_native/ValidationUtils.h',
                           'src/webnn_native/ValidationUtils_autogen.h',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/ValidationUtils.cpp',
                           'src/webnn_native/ValidationUtils_autogen.cpp',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/webnn_structs.h',
                           'src/webnn_native/webnn_structs_autogen.h',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/webnn_structs.cpp',
                           'src/webnn_native/webnn_structs_autogen.cpp',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/ProcTable.cpp',
                           'src/webnn_native/ProcTable.cpp', frontend_params))

        return renders

    def get_dependencies(self, args):
        deps = [os.path.abspath(args.webnn_json)]
        return deps


if __name__ == '__main__':
    sys.exit(run_generator(MultiGeneratorFromWebnnJSON()))
