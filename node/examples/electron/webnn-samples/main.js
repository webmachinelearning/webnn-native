// Modules to control application life and create native browser window
const {app, BrowserWindow} = require('electron')
const path = require('path')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow = {}

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 840,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableBlinkFeatures: 'SharedArrayBuffer',
      preload: path.join(__dirname, 'node_setup.js')
    }
  })


  let url = `file://${__dirname}/webnn-samples/index.html`

  const sampleNames = [
    "code",
    "image_classification",
    "lenet",
    "nsnet2",
    "object_detection",
    "rnnoise",
    "semantic_segmentation",
    "style_transfer",
  ]

  for (let argv of process.argv) {
    const argvName = argv.split('=')[0];
    const argvValue = argv.split('=')[1];
    if (argvName === "sample" && sampleNames.includes(argvValue)) {
      // Directly enter the entry of specified sample via '--sample' option.
      url = url.replace('index.html', `${argvValue}/index.html`);
      break
    }
  }

  for (let argv of process.argv) {
    if (argv.startsWith("numRuns=")) {
      // Load the index.html with 'numRuns' to run inference multiple times.
      url += '?' + argv
      break
    }
  }

  mainWindow.loadURL(url)

  // Emitted when the window is closed.
  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function() {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') app.quit()
})

app.on(
    'activate',
    function() {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (mainWindow === null) createWindow()
    })

    // In this file you can include the rest of your app's specific main process
    // code. You can also put them in separate files and require them here.
