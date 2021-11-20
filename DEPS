use_relative_paths = True

gclient_gn_args_file = 'build/config/gclient_args.gni'

vars = {
  'chromium_git': 'https://chromium.googlesource.com',
  'dawn_git': 'https://dawn.googlesource.com',
  'github_git': 'https://github.com',

  'dawn_standalone': True,
}

deps = {
  # Dependencies required for tests.
  'node/third_party/webnn-polyfill': {
    'url': '{github_git}/webmachinelearning/webnn-polyfill.git@daec7b0353885dd35851f338fc830497238cc1c8'
  },
  'node/third_party/webnn-polyfill/test-data': {
    'url': '{github_git}/webmachinelearning/test-data.git@b6f1565fefc103705a6ff580067eae7bb9d3b351'
  },
  'node/third_party/webnn-samples': {
    'url': '{github_git}/webmachinelearning/webnn-samples.git@d987c797f78c5cc6d665e61bd224db5e2605b145'
  },
  'node/third_party/webnn-samples/test-data': {
    'url': '{github_git}/webmachinelearning/test-data.git@b6f1565fefc103705a6ff580067eae7bb9d3b351'
  },
  'third_party/stb': {
    'url': '{github_git}/nothings/stb@b42009b3b9d4ca35bc703f5310eedc74f584be58'
  },

  # Dependencies required for code generator and infrastructure code.
  'third_party/dawn': {
    'url': '{dawn_git}/dawn.git@bf1c0cf52377b4db2bf3a433dc5056620aad7cdd'
  },

  # Dependencies required for backends.
  'third_party/DirectML': {
    'url': '{github_git}/microsoft/DirectML.git@af53ef743559079bd2ebb1aa83d9f98e87d774e1',
    'condition': 'checkout_win',
  },
  # GPGMM support for fast DML allocation and residency management.
  'third_party/gpgmm': {
    'url': '{github_git}/intel/gpgmm.git@7c81fae56e30c60030cb0a2c53310723e5c085d6',
    'condition': 'checkout_win',
  },
  'third_party/oneDNN': {
    'url': '{github_git}/oneapi-src/oneDNN.git@4a129541fd4e67e6897072186ea2817a3154eddd',
  },
  'third_party/XNNPACK': {
    'url': '{github_git}/google/XNNPACK.git@60fc61373f21f0ad3164cc719de464f4b787dc04'
  },

  # Dependencies required to use GN/Clang in standalone
  'build': {
    'url': '{chromium_git}/chromium/src/build@3769c3b43c3804f9f7f14c6e37f545639fda2852',
    'condition': 'dawn_standalone',
  },
  'buildtools': {
    'url': '{chromium_git}/chromium/src/buildtools@235cfe435ca5a9826569ee4ef603e226216bd768',
    'condition': 'dawn_standalone',
  },
  'tools/clang': {
    'url': '{chromium_git}/chromium/src/tools/clang@b12d1c836e2bb21b966bf86f7245bab9d257bb6b',
    'condition': 'dawn_standalone',
  },
  'tools/clang/dsymutil': {
    'packages': [
      {
        'package': 'chromium/llvm-build-tools/dsymutil',
        'version': 'M56jPzDv1620Rnm__jTMYS62Zi8rxHVq7yw0qeBFEgkC',
      }
    ],
    'condition': 'checkout_mac or checkout_ios',
    'dep_type': 'cipd',
  },

  # Testing, GTest and GMock
  'testing': {
    'url': '{chromium_git}/chromium/src/testing@3e2640a325dc34ec3d9cb2802b8da874aecaf52d',
    'condition': 'dawn_standalone',
  },
  'third_party/googletest': {
    'url': '{chromium_git}/external/github.com/google/googletest@2828773179fa425ee406df61890a150577178ea2',
    'condition': 'dawn_standalone',
  },

  # Jinja2 and MarkupSafe for the code generator
  'third_party/jinja2': {
    'url': '{chromium_git}/chromium/src/third_party/jinja2@a82a4944a7f2496639f34a89c9923be5908b80aa',
    'condition': 'dawn_standalone',
  },
  'third_party/markupsafe': {
    'url': '{chromium_git}/chromium/src/third_party/markupsafe@0944e71f4b2cb9a871bcbe353f95e889b64a611a',
    'condition': 'dawn_standalone',
  },

}

hooks = [
  # Pull the compilers and system libraries for hermetic builds
  {
    'name': 'sysroot_x86',
    'pattern': '.',
    'condition': 'checkout_linux and ((checkout_x86 or checkout_x64) and dawn_standalone)',
    'action': ['python', 'build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x86'],
  },
  {
    'name': 'sysroot_x64',
    'pattern': '.',
    'condition': 'checkout_linux and (checkout_x64 and dawn_standalone)',
    'action': ['python', 'build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x64'],
  },
  {
    # Update the Mac toolchain if possible, this makes builders use "hermetic XCode" which is
    # is more consistent (only changes when rolling build/) and is cached.
    'name': 'mac_toolchain',
    'pattern': '.',
    'condition': 'checkout_mac',
    'action': ['python', 'build/mac_toolchain.py'],
  },
  {
    # Update the Windows toolchain if necessary. Must run before 'clang' below.
    'name': 'win_toolchain',
    'pattern': '.',
    'condition': 'checkout_win and dawn_standalone',
    'action': ['python', 'build/vs_toolchain.py', 'update', '--force'],
  },
  {
    # Note: On Win, this should run after win_toolchain, as it may use it.
    'name': 'clang',
    'pattern': '.',
    'action': ['python', 'tools/clang/scripts/update.py'],
    'condition': 'dawn_standalone',
  },
  {
    # Pull rc binaries using checked-in hashes.
    'name': 'rc_win',
    'pattern': '.',
    'condition': 'checkout_win and (host_os == "win" and dawn_standalone)',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--no_auth',
                '--bucket', 'chromium-browser-clang/rc',
                '-s', 'build/toolchain/win/rc/win/rc.exe.sha1',
    ],
  },
  # Pull clang-format binaries using checked-in hashes.
  {
    'name': 'clang_format_win',
    'pattern': '.',
    'condition': 'host_os == "win"',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'buildtools/win/clang-format.exe.sha1',
    ],
  },
  {
    'name': 'clang_format_mac',
    'pattern': '.',
    'condition': 'host_os == "mac"',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'buildtools/mac/clang-format.sha1',
    ],
  },
  {
    'name': 'clang_format_linux',
    'pattern': '.',
    'condition': 'host_os == "linux"',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'buildtools/linux64/clang-format.sha1',
    ],
  },
  # Update build/util/LASTCHANGE.
  {
    'name': 'lastchange',
    'pattern': '.',
    'condition': 'dawn_standalone',
    'action': ['python', 'build/util/lastchange.py',
               '-o', 'build/util/LASTCHANGE'],
  },
  {
    # Download the DirectML NuGet package.
    'name': 'download_dml_unpkg',
    'pattern': '.',
    'condition': 'checkout_win',
    'action': ['python3', 'src/webnn_native/dml/deps/script/download_dml.py'],
  }
]

recursedeps = [
  # buildtools provides clang_format, libc++, and libc++abi
  'buildtools',
]
