const express = require('express');
const upload = require('express-fileupload');


const app = express();

app.use(upload('uploads'))


app.get('/', (req,res) => {
    res.sendFile(__dirname +  '/index.html'  )
    
})


app.post('/', (req, res) => {
    if (req.files){
        console.log(req.files)
        var file  = req.files.file
        var filename  = file.name  
        console.log(filename)
        file.mv('./uploads/' +  filename, function(err) {
            if(err){
                res.send(err)
            }else{
                res.send("Upload Successful!")
            }
        } )
    }
})

app.listen(8000)