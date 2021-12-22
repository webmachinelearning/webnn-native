const fse = require('fs-extra');

module.exports = {
  packagerConfig: {
    asar: {
      unpack: "node_setup.js",
      unpackDir: "{lib,build}"
   },
  },
  makers: [
    {
      name: '@electron-forge/maker-zip'
    }
   ],
  hooks: {
    generateAssets: async () => {
      // Copy WebNN libs, source code and dependences of Webnn-samples
      const node_path = '../../../';
      const copy_list = [
        {"from": `${node_path}node_setup.js`, "to": 'node_setup.js'},
        {"from": `${node_path}lib`, "to": 'lib'},
        {"from": `${node_path}build`, "to": 'build'},
        {"from": `${node_path}third_party/webnn-samples`, "to": 'webnn-samples'},
        {"from": `${node_path}third_party/webnn-samples/rnnoise/process/process.wasm`, "to": 'webnn-samples/rnnoise/process.wasm'},
      ];
     for (let copy of copy_list) {
      await fse.copy(copy.from, copy.to, {overwrite: true});
     }

      console.log('Copy WebNN libs and samples successfully!');
    }
  }
}
