const express = require("express")
const multer = require("multer")
const cors = require("cors")
const bodyParser = require("body-parser")

const { spawn } = require("child_process")
const path = require("path")

// Initialize Express app
const app = express()
app.use(cors())
// Middleware to parse JSON in request body
app.use(express.json())
app.use(bodyParser.json())

const PORT = process.env.PORT || 3000
app.use(express.static("../frontend/public"))

// Set up storage for uploaded images
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.resolve(__dirname, "uploads")) // Save files to 'uploads' folder
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname)) // Use timestamp as filename
  },
})

const upload = multer({ storage: storage })

// POST route to handle the image upload
app.post("/upload", upload.single("image"), (req, res) => {
  // Log the entire req.file object to debug
  // console.log("Uploaded file: ", req.file)
  // const filePath = path.join(__dirname, req.file.path) // Path to the uploaded file
  const filePath = req.file.path // Path to the uploaded file
  console.log("filepath: ", filePath)
  // Use absolute path for Python script
  //const pythonScriptPath = path.resolve(__dirname, "process_image.py") //test
  const pythonScriptPath = path.resolve(__dirname, "../python-scripts/main.py")
  console.log("pythonScriptPath: ", pythonScriptPath)

  // Call the Python script, passing the image file path as an argument
  const pythonProcess = spawn("python", [pythonScriptPath, filePath])

  // Capture the output from the Python script
  let result = ""
  pythonProcess.stdout.on("data", (data) => {
    result += data.toString()
  })

  pythonProcess.stderr.on("data", (data) => {
    console.error(`Error: ${data}`)
  })

  pythonProcess.on("close", (code) => {
    if (code === 0) {
      res.json({ result: result }) // Send the result back to the frontend
    } else {
      res.status(500).json({ result: "Error processing the image" })
    }
  })
})

// POST route to handle JSON content and run a Python script
app.post("/fetchisbn", (req, res) => {
  const content = req.body.content // JSON content from the frontend

  if (!content) {
    return res
      .status(400)
      .json({ error: "Content(author & title) is required" })
  }

  const pythonScriptPath = path.resolve(
    __dirname,
    "../python-scripts/fetch_book_info.py"
  )

  console.log("Running Python script with content:", content) //ok

  // Spawn Python process with JSON content as argument
  const pythonProcess = spawn("python", [
    pythonScriptPath,
    content.title,
    content.author,
  ])

  let result = ""
  console.log("Python Script Path:", pythonScriptPath) //ok
  //Capture the output from the Python script
  pythonProcess.stdout.on("data", (data) => {
    result += data.toString()
  })

  pythonProcess.stderr.on("data", (data) => {
    console.error(`Error: ${data.toString()}`)
  })

  pythonProcess.on("close", (code) => {
    if (code === 0) {
      res.json({ result: result }) // Send the result back to the frontend
    } else {
      res.status(500).json({ error: "Error fetching isbn." })
    }
  })
})
app.post("/search", (req, res) => {
  const content = req.body // JSON content from the frontend
  console.log(content)

  if (!content) {
    return res
      .status(400)
      .json({ error: "Content(author & title) is required" })
  }

  const pythonScriptPath = path.resolve(
    __dirname,
    "../check_book_existence/book_exists.py"
  )

  console.log(
    "Running Python script to search in database with content:",
    content.isbn10
  ) //ok

  // Spawn Python process with JSON content as argument
  const pythonProcess = spawn("python", [
    pythonScriptPath,
    content.isbn10,
    content.isbn13,
  ])

  let result = ""
  console.log("Python Script Path:", pythonScriptPath) //ok
  //Capture the output from the Python script
  pythonProcess.stdout.on("data", (data) => {
    result += data.toString()
  })

  pythonProcess.stderr.on("data", (data) => {
    console.error(`Error: ${data.toString()}`)
  })

  pythonProcess.on("close", (code) => {
    if (code === 0) {
      res.json({ result: result }) // Send the result back to the frontend
    } else {
      res.status(500).json({ error: "Error fetching isbn." })
    }
  })
})
// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`)
})
