const express = require('express');
const app = express();
const PORT = 3000;

app.use(express.static('public'));

app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.post('/', (req, res) => {
    if (req.files){
        console.log(req.files)
        var file  = req.files.file
        var filename  = file.name  
        console.log(filename)
        file.mv('./public/' +  filename, function(err) {
            if(err){
                res.send(err)
            }else{
                res.send("Upload Successful!")
            }
        } )
    }
})


app.listen(PORT, () => console.log(`Server listening on port: ${PORT}`));