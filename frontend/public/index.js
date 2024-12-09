// Function to handle the image upload
function uploadImage() {
  console.log("clicked")
  const fileInput = document.getElementById("imageInput")
  const file = fileInput.files[0]
  console.log(file)

  if (!file) {
    alert("Please select an image first!")
    return
  }

  // Create a FormData object to send the file to the backend
  const formData = new FormData()
  //key = image
  formData.append("image", file)

  // Send the image file to the backend using fetch
  fetch("http://localhost:3000/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("sever response: ", data)
      document.getElementById("upload_result").textContent = data.result // Show result in the frontend
    })
    .catch((error) => {
      console.error("Error:", error)
    })
}

async function getBookISBN() {
  try {
    // Get the content from <pre id="upload_result">, json obj
    const uploadContent = JSON.parse(
      document.getElementById("upload_result").textContent
    )
    console.log("uploadContent: ", uploadContent) //ok
    // Send the content to the backend
    const response = await fetch("/fetchisbn", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ content: uploadContent }),
    })
      .then((response) => response.json())
      .then((data) => {
        document.getElementById("fetch_result").textContent = data.result

        console.log("sever response: ", data.result)
      })
  } catch (error) {
    console.error("Error:", error)
    document.getElementById("fetch_result").textContent =
      "Error in fetching isbn."
  }
}

function search() {
  const isbn10 = document.getElementById("isbn10").value
  const isbn13 = document.getElementById("isbn13").value

  // Check if either ISBN-10 or ISBN-13 is provided
  if (isbn10 || isbn13) {
    const requestData = {
      isbn10: isbn10,
      isbn13: isbn13,
    }

    // Send POST request to the backend
    fetch("/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestData),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display search result
        document.getElementById("search_result").textContent = data.result
        console.log("data result in search response: ", data)
      })
      .catch((error) => {
        document.getElementById("search_result").innerHTML =
          "Error occurred while searching data."
      })
  } else {
    alert("Please enter either ISBN-10 or ISBN-13.")
  }
}
