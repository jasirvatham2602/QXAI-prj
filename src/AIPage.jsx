import { useState } from 'react';

function AIPage() {
    const [image, setImage] = useState(null); 
    const [preview, setPreview] = useState(null); 
    const [file, setFile] = useState(null); 
    const [res, setRes] = useState(null); 
    const [saliencyMapsPath, setSaliencyMapsPath] = useState(null); 
    // const onImageChange = (event) => {
    //     if (event.target.files && event.target.files[0]) {
    //         setImage(event.target.files[0]);
    //         setPreview(URL.createObjectURL(event.target.files[0]))
    //     }
    //     console.log(image);
    //     console.log(preview);
    // }
    let previewImage = (event) => {
        let reader = new FileReader(); 
        let file = event.target.files[0]; 
        let imageBase64; 
        if (!file) {
            setFile(null);
            setPreview(null); 
            return;
        }
        reader.onload = () => {
            document.getElementById("preview").src = reader.result
            console.log(reader.result);
            imageBase64 = reader.result; 
        }; 
        reader.readAsDataURL(event.target.files[0]);
        setFile(file); 
        setPreview(URL.createObjectURL(file)); 
        console.log("file");
        console.log(file);
        console.log("preview");
        console.log(preview);
    } 
    let handleSubmit = async (event) => {
        event.preventDefault();  
        if (!file) {
            alert("Please upload an image"); 
            return; 
        }
        const formData = new FormData(); 
        formData.append('image', file);
        // formData.append('link', preview);
        // console.log(formData); 
 
        const res = await fetch("https://jasirvatham2602-qxai-backend-hugging-face.hf.space/predict", {
            method: 'POST',
            body: formData, 
        }); 
        const data = await res.json(); 
        console.log('data');
        console.log(data);
        setRes(data); 
        // upload data to backend here 
    };
    return (
         <>
            <div className="container" style={{display: 'block'}}>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="exampleInputEmail1">Email address</label>
                    </div> 
                    <div className="form-group form-check">
                        <input type="checkbox" className="form-check-input" id="exampleCheck1"/>
                        <label className="form-check-label" htmlFor="exampleCheck1">Check me out</label>
                    </div>
                    <img id='preview' width="200" src="" alt="" /> <br />
                    <input type="file" onChange={previewImage}/>
                    <button type= "submit" className="btn btn-primary">Upload</button>
                </form>
                <br />
                    {res && (
                    <pre>{JSON.stringify(res, null, 2)}</pre>
                      )}
                <br />
              <div className='SaliencyMaps' style={{display: 'block'}}>
                      <img src={res == null ? "" : res.saliency_map_url} style={{width: 1000, display: 'block'}} alt="" />
              </div>
            </div>
        </>
    );
}
export default AIPage;