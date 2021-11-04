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
      // Copy WebNN libs, source code and dependences of RNNoise sample
      const node_path = '../../../';
      const samples_path = node_path + 'third_party/webnn-samples/';
      const copy_list = [
        {"from": `${node_path}node_setup.js`, "to": 'node_setup.js'},
        {"from": `${node_path}lib`, "to": 'lib'},
        {"from": `${node_path}build`, "to": 'build'},
        {"from": `${samples_path}common`, "to": 'common'},
        {"from": `${samples_path}test-data/models/rnnoise`,
         "to": 'test-data/models/rnnoise'},
        {"from": `${samples_path}rnnoise`, "to": 'rnnoise'},
        {"from": `${samples_path}rnnoise/process/process.wasm`,
         "to": 'rnnoise/process.wasm'},
      ];
     for (let copy of copy_list) {
      await fse.copy(copy.from, copy.to, {overwrite: true});
     }

      console.log('Copy WebNN libs and samples successfully!');
    }
  }
}
