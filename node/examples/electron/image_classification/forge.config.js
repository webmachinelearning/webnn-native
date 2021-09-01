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
      // Copy WebNN libs, source code and dependences of image classification sample
      const node_path = '../../../';
      const samples_path = node_path + 'third_party/webnn-samples/';
      const copy_list = [
        {"from": `${node_path}node_setup.js`, "to": 'node_setup.js'},
        {"from": `${node_path}lib`, "to": 'lib'},
        {"from": `${node_path}build`, "to": 'build'},
        {"from": `${samples_path}common`, "to": 'common'},
        {"from": `${samples_path}test-data/models/mobilenetv2_nhwc`,
         "to": 'test-data/models/mobilenetv2_nhwc'},
        {"from": `${samples_path}test-data/models/mobilenetv2_nchw`,
         "to": 'test-data/models/mobilenetv2_nchw'},
        {"from": `${samples_path}test-data/models/resnet101v2_nhwc`,
         "to": 'test-data/models/resnet101v2_nhwc'},
        {"from": `${samples_path}test-data/models/resnet50v2_nchw`,
         "to": 'test-data/models/resnet50v2_nchw'},
        {"from": `${samples_path}test-data/models/squeezenet1.0_nhwc`,
         "to": 'test-data/models/squeezenet1.0_nhwc'},
        {"from": `${samples_path}test-data/models/squeezenet1.1_nchw`,
         "to": 'test-data/models/squeezenet1.1_nchw'},
        {"from": `${samples_path}image_classification`,
         "to": 'image_classification'},
      ];
     for (let copy of copy_list) {
      await fse.copy(copy.from, copy.to, {overwrite: true});
     }

      console.log('Copy WebNN libs and samples successfully!');
    }
  }
}
