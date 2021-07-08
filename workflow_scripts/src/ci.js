'use strict';

const {execFile, spawn} = require('child_process');
const compressing = require('compressing');
const fs = require('fs-extra');
const os = require('os');
const path = require('path');
const urllib = require('urllib');

const backend = process.argv.slice(2)[0];
const cwd = process.cwd();
const rootDir = path.join(cwd, '..');
const outDir = path.join(rootDir, 'out', 'Release');
const depotToolsDir = path.join(cwd, 'depot_tools');

const childProcessSpawn = async (cmd, args, cwd) => {
  console.log(`Execute commad: ${cmd} ${[...args].join(' ')}`);
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, [...args], {cwd: cwd, shell: true});
    child.stdout.on('data', (data) => {
      console.log(data.toString());
    });
    child.stderr.on('data', (data) => {
      console.log(data.toString());
    });
    child.on('close', (code) => {
      resolve(code);
    });
  });
};

const installDepotTools = async () => {
  if (os.platform() == 'win32') {
    await urllib.request('https://storage.googleapis.com/chrome-infra/depot_tools.zip', {
      streaming: true,
      followRedirect: true,
    })
        .then((result) => compressing.zip.uncompress(result.res, depotToolsDir))
        .then(() => console.log('Install depot_tools: SUCCESS'))
        .catch((error) => console.log(`Install depot_tools: FAIL\n ${error}`));
  } else {
    await childProcessSpawn(
        'git',
        ['clone', 'https://chromium.googlesource.com/chromium/tools/depot_tools.git', depotToolsDir],
        rootDir)
        .then((code) =>{
          if (code == 0) {
            console.log('Install depot_tools: SUCCESS');
          } else {
            console.log('Install depot_tools: FAIL');
            process.exit(1);
          }
        });
  }

  const separator = os.platform() == 'win32' ? ';' : ':';
  process.env['PATH'] = depotToolsDir + separator + process.env['PATH'];

  if (os.platform() == 'win32') {
    process.env['DEPOT_TOOLS_WIN_TOOLCHAIN'] = '0';
  }
};

const syncCode = async () => {
  // gclient sync
  fs.copyFileSync(path.join(rootDir, 'scripts', 'standalone.gclient'),
      path.join(rootDir, '.gclient'));

  process.chdir(rootDir);
  await childProcessSpawn(
    os.platform() == 'win32' ? 'gclient.bat' : 'gclient',
    ['sync'], rootDir)
      .then((code) =>{
        if (code == 0) {
          console.log('Run "gclient sync": SUCCESS');
        } else {
          console.log('Run "gclient sync": FAIL');
          process.exit(1);
        }
      });
};

const buildTest = async () => {
  // gn gen
  await childProcessSpawn(
    os.platform() == 'win32' ? 'gn.bat' : 'gn',
    ['gen', outDir, `--args="is_debug=false webnn_enable_${backend}=true"`],
    rootDir)
      .then((code) =>{
        if (code == 0) {
          console.log(
              `Run "gn gen ${outDir} --args=` +
              `"is_debug=false webnn_enable_${backend}=true"": SUCCESS`);
        } else {
          console.log(
              `Run "gn gen ${outDir} --args=` +
              `"is_debug=false webnn_enable_${backend}=true"": FAIL`);
          process.exit(1);
        }
      });

  // ninja
  await childProcessSpawn('ninja', ['-C', outDir], rootDir)
      .then((code) =>{
        if (code == 0) {
          console.log(
              `Run "ninja -C ${outDir}": SUCCESS`);
        } else {
          console.log(
              `Run "ninja -C ${outDir}": FAIL`);
          process.exit(1);
        }
      });

  const outNodeDir = path.join(outDir, 'node');
  fs.mkdirpSync(outNodeDir);
  fs.copySync(path.join(rootDir, 'node'), outNodeDir);

  process.chdir(outDir);

  // Run Unit Tests
  execFile(
    os.platform() == 'win32' ? 'webnn_unittests.exe' : './webnn_unittests',
    (err, data) => {
      if (err) {
        console.log(`Run Unit Tests: FAIL\n${err}\n${data}`);
      } else {
        console.log(`Run Unit Tests: SUCCESS\n${data}`);
      }
    });

  if (backend != 'null') {
    // Run End2End Tests
    execFile('webnn_end2end_tests.exe', (err, data) => {
      if (err) {
        console.log(`Run End2End Tests: FAIL\n${err}\n${data}`);
      } else {
        console.log(`Run End2End Tests: SUCCESS\n${data}`);
      }
    });
  }
};

(async () => {
  await installDepotTools();
  await syncCode();
  await buildTest();
})().catch((error) => console.error(error));
