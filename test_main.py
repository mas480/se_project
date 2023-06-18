from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
    
    
def test_read_predict_positive():
    response = client.post("/predict/",
        json={"text": "Самый большой город"}
    )
    json_data = response.json() 

    assert response.status_code == 200
    assert json_data == 'Самый большой город в мире, столица штата Нью-Йорк, город, в котором сосредоточены основные'