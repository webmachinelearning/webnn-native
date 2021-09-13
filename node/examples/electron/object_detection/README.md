# An Electron.js example of Object Detection using webnn-native

### Install

Firstly, ensure that you have done these steps in [README.md](/node/README.md), then run:
```bash
npm install
```

### Run

```bash
npm start
```

### Run with parameters

- `numRuns`: typeof `number`, used for runing inference multiple times to get the median inference time, e.g.:
```bash
npm start numRuns=100
```
- `device`: one of `["default", "gpu", "cpu"]`, used for setting preferred kind of device used, e.g.:
```bash
npm start device=gpu
```

### Package

Bundles source code with a renamed Electron executable and supporting files into `out` folder ready for distribution.

```bash
npm run package
```

### Distribution

Creates a distributable using Electron Forge's `make` command:

```bash
npm run make
```

### Screenshot

![screenshot](screenshot.png)