use_relative_paths = True

gclient_gn_args_file = 'build/config/gclient_args.gni'

gclient_gn_args = [
  'generate_location_tags',
]

vars = {
  'chromium_git': 'https://chromium.googlesource.com',
  'dawn_git': 'https://dawn.googlesource.com',
  'github_git': 'https://github.com',

  'dawn_standalone': True,
  'checkout_onnxruntime': True,
  'checkout_polyfill': True,
  'checkout_samples': True,

  'dawn_gn_version': 'git_revision:fc295f3ac7ca4fe7acc6cb5fb052d22909ef3a8f',
  # GN variable required by //testing that will be output in the gclient_args.gni
  'generate_location_tags': False
}

deps = {
  # Dependencies required for tests.
  'node/third_party/webnn-polyfill': {
    'url': '{github_git}/webmachinelearning/webnn-polyfill.git@6a9754420d9c761cc8d9ecb7625b061d849bca6f',
    'condition': 'checkout_polyfill',
  },
  'node/third_party/webnn-polyfill/test-data': {
    'url': '{github_git}/webmachinelearning/test-data.git@b6f1565fefc103705a6ff580067eae7bb9d3b351',
    'condition': 'checkout_polyfill',
  },
  'node/third_party/webnn-samples': {
    'url': '{github_git}/webmachinelearning/webnn-samples.git@7e77194153fe87d7d20f3cf4b1545429fb8392b9',
    'condition': 'checkout_samples'
  },
  'node/third_party/webnn-samples/test-data': {
    'url': '{github_git}/webmachinelearning/test-data.git@d9f1537096b0fcbbef0889b540bf8ce6c8833969',
    'condition': 'checkout_samples'
  },
  'third_party/stb': {
    'url': '{github_git}/nothings/stb@af1a5bc352164740c1cc1354942b1c6b72eacb8a'
  },

  # Dependencies required for code generator and infrastructure code.
  'third_party/dawn': {
    'url': '{dawn_git}/dawn.git@bf1c0cf52377b4db2bf3a433dc5056620aad7cdd'
  },

  # Dependencies required for backends.
  'third_party/DirectML': {
    'url': '{github_git}/microsoft/DirectML.git@c3f16a701beeeefc9ce5b67c71b554a6903c0f67',
    'condition': 'checkout_win',
  },
  # GPGMM support for fast DML allocation and residency management.
  'third_party/gpgmm': {
    'url': '{github_git}/intel/gpgmm.git@61fcfcbd872b7643423ac8c9555f3a6a366904ee',
    'condition': 'checkout_win',
  },
  'third_party/oneDNN': {
    'url': '{github_git}/oneapi-src/oneDNN.git@4a129541fd4e67e6897072186ea2817a3154eddd',
  },
  'third_party/XNNPACK': {
    'url': '{github_git}/google/XNNPACK.git@42806cdefa7c48247b640a43024040c735d97f29'
  },
  'third_party/onnxruntime': {
    'url': '{github_git}/microsoft/onnxruntime.git@0d9030e79888d1d5828730b254fedc53c7b640c1',
    'condition': 'checkout_onnxruntime',
  },

  # Dependencies required to use GN/Clang in standalone
  'build': {
    'url': '{chromium_git}/chromium/src/build@555c8b467c21e2c4b22d00e87e3faa0431df9ac2',
    'condition': 'dawn_standalone',
  },
  'buildtools': {
    'url': '{chromium_git}/chromium/src/buildtools@f78b4b9f33bd8ef9944d5ce643daff1c31880189',
    'condition': 'dawn_standalone',
  },
  'buildtools/linux64': {
    'packages': [{
      'package': 'gn/gn/linux-amd64',
      'version': Var('dawn_gn_version'),
    }],
    'dep_type': 'cipd',
    'condition': 'dawn_standalone and host_os == "linux"',
  },
  'buildtools/win': {
    'packages': [{
      'package': 'gn/gn/windows-amd64',
      'version': Var('dawn_gn_version'),
    }],
    'dep_type': 'cipd',
    'condition': 'dawn_standalone and host_os == "win"',
  },
  'buildtools/third_party/libc++/trunk': {
    'url': '{chromium_git}/external/github.com/llvm/llvm-project/libcxx.git@79a2e924d96e2fc1e4b937c42efd08898fa472d7',
    'condition': 'dawn_standalone',
  },

  'buildtools/third_party/libc++abi/trunk': {
    'url': '{chromium_git}/external/github.com/llvm/llvm-project/libcxxabi.git@2715a6c0de8dac4c7674934a6b3d30ba0c685271',
    'condition': 'dawn_standalone',
  },

  'tools/clang': {
    'url': '{chromium_git}/chromium/src/tools/clang@8b7330592cb85ba09505a6be7bacabd0ad6160a3',
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
    'url': '{chromium_git}/chromium/src/testing@d485ae97b7900c1fb7edfbe2901ae5adcb120865',
    'condition': 'dawn_standalone',
  },
  'third_party/googletest': {
    'url': '{chromium_git}/external/github.com/google/googletest@6b74da4757a549563d7c37c8fae3e704662a043b',
    'condition': 'dawn_standalone',
  },
  # This is a dependency of //testing
  'third_party/catapult': {
    'url': '{chromium_git}/catapult.git@fa35beefb3429605035f98211ddb8750dee6a13d',
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
    'action': ['python3', 'build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x86'],
  },
  {
    'name': 'sysroot_x64',
    'pattern': '.',
    'condition': 'checkout_linux and (checkout_x64 and dawn_standalone)',
    'action': ['python3', 'build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x64'],
  },
  {
    # Update the Mac toolchain if possible, this makes builders use "hermetic XCode" which is
    # is more consistent (only changes when rolling build/) and is cached.
    'name': 'mac_toolchain',
    'pattern': '.',
    'condition': 'checkout_mac',
    'action': ['python3', 'build/mac_toolchain.py'],
  },
  {
    # Update the Windows toolchain if necessary. Must run before 'clang' below.
    'name': 'win_toolchain',
    'pattern': '.',
    'condition': 'checkout_win and dawn_standalone',
    'action': ['python3', 'build/vs_toolchain.py', 'update', '--force'],
  },
  {
    # Note: On Win, this should run after win_toolchain, as it may use it.
    'name': 'clang',
    'pattern': '.',
    'action': ['python3', 'tools/clang/scripts/update.py'],
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
    'condition': 'host_os == "win" and dawn_standalone',
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
    'action': ['python3', 'build/util/lastchange.py',
               '-o', 'build/util/LASTCHANGE'],
  },
  {
    # Download the DirectML NuGet package.
    'name': 'download_dml_unpkg',
    'pattern': '.',
    'condition': 'checkout_win',
    'action': ['python3', 'third_party/scripts/download_dml.py'],
  }
]

recursedeps = [
  # buildtools provides clang_format, libc++, and libc++abi
  'buildtools',
]
