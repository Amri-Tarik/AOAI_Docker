const express = require("express");
const app = express();
var cors = require('cors')
const port = 3000;
const ID = "admin"

var corsOptions = {
  origin: 'https://int-aoai.netlify.app'
}

app.use(express.json())
app.use(cors(corsOptions))

let questions = [
			{question : "first question ?", answer : ["first answer","second answer"] },
			{question : "Second question ?", answer : ["first answer","second answer","third answer"] },
			{question : "third question ?", answer : ["first answer","second answer","third answer"] },
			{question : "fourth question ?", answer : ["first answer","second answer"] }
		]

app.post("/auth", (req, res) => {
	const spawn = require("child_process").spawn;
	let reply = '';
	let e = 0
	const pythonProcess = spawn('python3', ['app.py', req.body.id.toLowerCase()]);
	pythonProcess.stdout.on('data', (data) => {
		reply = JSON.parse(data.toString())
	})
	pythonProcess.stderr.on('data', (err) => {
		e = 1
		console.log(err.toString())
	})
	pythonProcess.on('exit', (code) => {
		if (e == 0) {
			res.send({
				validation : "ok",
				gm : reply.gm,
				gi : reply.gi,
				ge : reply.ge,
			})
		} else {
			res.send({validation : "la27"})
		}
		
	});
});

app.post("/gm", (req, res) => {
	if(req.body.id.toLowerCase() == ID) {
		// retrive data from the database
		setTimeout(()=>{res.send({
			form : "ok",
			l : questions.length,
			list : questions,
		})},1000)
	} else {
		res.send({validation : "la27"})
	}
});

app.post("/gi", (req, res) => {
	if(req.body.id.toLowerCase() == ID) {
		// retrive data from the database
		setTimeout(()=>{res.send({
			form : "ok",
			l : questions.length,
			list : questions,
		})},1000)
	} else {
		res.send({validation : "la27"})
	}
});

app.post("/ge", (req, res) => {
	if(req.body.id.toLowerCase() == ID) {
		// retrive data from the database
		setTimeout(()=>{res.send({
			form : "ok",
			l : questions.length,
			list : questions,
		})},1000)
	} else {
		res.send({validation : "la27"})
	}
});

// app.post("/python", (req, res) => {
// 	console.log("request received")
// 	const spawn = require("child_process").spawn;
// 	let reply = '';
// 	const pythonProcess = spawn('python3', ['example.py', 'tfa7']);
// 	pythonProcess.stdout.on('data', (data) => {
// 		reply = JSON.parse(data.toString()).reply
// 		console.log("total message is : " + data.toString());
// 	})
// 	pythonProcess.on('error', (err) => {
// 		console.error('Failed to start subprocess.');
// 	})
// 	pythonProcess.on('exit', (code) => {
// 		console.log('reply is : ' + reply);
// 		res.send(reply);
// 	});
// });

app.get("/", (req, res) => {
 	res.send("ok");
});

app.listen(port, () => {
  	console.log(`Example app listening at http://localhost:${port}`);
});
