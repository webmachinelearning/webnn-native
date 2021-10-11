{
  'variables': {
    'WEBNN_NATIVE_DIR': '<@(module_root_dir)/../',
    'WEBNN_NATIVE_LIB_PATH': '<@(module_root_dir)/../<(webnn_native_lib_path)',
  },
  'targets': [
    {
      'target_name': 'webnn_node',
      'sources': [
        "<!@(node -p \"require('fs').readdirSync('./src').map(f=>'src/'+f).join(' ')\")",
        "<!@(node -p \"require('fs').readdirSync('./src/ops').map(f=>'src/ops/'+f).join(' ')\")",
        # webnn_cpp.cpp must be relative path.
        '../<(webnn_native_lib_path)/gen/src/webnn/webnn_cpp.cpp',
      ],
      'cflags!': [ '-fno-exceptions', '-fno-rtti'],
      'cflags_cc!': [ '-fno-exceptions', '-fno-rtti'],
      'default_configuration': 'Release',
      'configurations': {
        'Debug': {
          'msvs_settings': {
            'VCCLCompilerTool': {
              'ExceptionHandling': 1,
              'RuntimeTypeInfo': 'true',
              'RuntimeLibrary': 3 # MultiThreadedDebugDLL (/MDd)
            },
          },
        },
        'Release': {
          'msvs_settings': {
            'VCCLCompilerTool': { 
              'ExceptionHandling': 1,
              'RuntimeTypeInfo': 'true',
              'RuntimeLibrary': 2 # MultiThreadedDLL (/MD)
            },
          },
        }
      },
      'include_dirs' : [
        '<!@(node -p "require(\'node-addon-api\').include")',
        '<(module_root_dir)/src',
        '<(WEBNN_NATIVE_DIR)/src/include',
        '<(WEBNN_NATIVE_LIB_PATH)/gen/src/include',
        '<(WEBNN_NATIVE_LIB_PATH)/../../src/include',
      ],
      'library_dirs' : [
        '<(WEBNN_NATIVE_LIB_PATH)',
      ],
      'conditions': [
        [ 'OS=="win"', {
            'libraries' : [
              '-lwebnn_native.dll.lib',
              '-lwebnn_proc.dll.lib'
            ],
            'copies': [ {
              'destination': '<(module_root_dir)/build/$(Configuration)/',
              'files': [
                  '<(WEBNN_NATIVE_LIB_PATH)/libc++.dll',
                  '<(WEBNN_NATIVE_LIB_PATH)/webnn_native.dll',
                  '<(WEBNN_NATIVE_LIB_PATH)/webnn_proc.dll',
                  '<(WEBNN_NATIVE_LIB_PATH)/DirectML.dll',
                  '<(WEBNN_NATIVE_LIB_PATH)/ngraph_c_api.dll',
              ]
            } ]
          },
          'OS=="linux"', {
            'libraries' : [
              "-Wl,-rpath,'$$ORIGIN'/..",
              '-lwebnn_native',
              '-lwebnn_proc'
            ],
            'copies': [ {
              'destination': '<(module_root_dir)/build/$(Configuration)/',
              'files': [
                  '<(WEBNN_NATIVE_LIB_PATH)/libwebnn_native.so',
                  '<(WEBNN_NATIVE_LIB_PATH)/libwebnn_proc.so',
                  '<(WEBNN_NATIVE_LIB_PATH)/libngraph_c_api.so',
              ]
            } ]
          }
        ]
      ]
    }
  ]
}
