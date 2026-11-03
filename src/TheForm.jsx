import { useState } from "react";

function TheForm() {
    let [username, setUsername] = useState("");
    let [password, setPassword] = useState("");
    let [isChecked, setIsChecked] = useState(false);

    let onChange = (event) => {

    }
    let handleSubmit = async (event) => {
        event.preventDefault();
        console.log(`Username: ${username} Password ${password} isChecked: ${isChecked}`);  
    }
    return (
        <>
       
            <h1> Password: {password}</h1>
            <h1> Username: {username}</h1>
            <div className="container">
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="exampleInputEmail1">Email address</label>
                        <input type="email" className="form-control" id="exampleInputEmail1" aria-describedby="emailHelp" placeholder="Enter email" onChange={(e) => setUsername(e.target.value)}/> 
                    </div>
                    <div className="form-group">
                        <label htmlFor="exampleInputPassword1">Password</label>
                        <input type="password" className="form-control" id="exampleInputPassword1" placeholder="Password" suggested="c"
                        onChange={(e) => setPassword(e.target.value)}
                        />
                    </div>
                    <div className="form-group form-check">
                        <input type="checkbox" className="form-check-input" id="exampleCheck1" onChange={(e) => setIsChecked(e.target.checked)}/>
                        <label className="form-check-label" htmlFor="exampleCheck1">Check me out</label>
                    </div>
                    <button type="submit" className="btn btn-primary">Submit</button>
                </form>
            </div>
        </>
    );
}
export default TheForm;