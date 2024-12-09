// convert original database into an array
// const database0 = [
//   {
//     title: "Qaqavii",
//     author: "Miriam Korner",
//     pubDate: "2019",
//     isbn10: ["0889955700"],
//     isbn13: ["9780889955707"],
//   },
//   {},
// ]
const fs = require("fs")

// Read the raw database file
const rawDatabase = fs.readFileSync("./bookMeta.jsonl.json", "utf8")

// Automatically fix: Wrap with [] and add commas
const fixedDatabase = `[${rawDatabase.replace(/}\s*{/g, "},{")}]`

// Save the fixed database to a new file
fs.writeFileSync("fixedDatabase.json", fixedDatabase)

console.log("Database has been fixed and saved as fixedDatabase.json")
