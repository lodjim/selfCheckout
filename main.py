from fastapi import FastAPI,File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from fastapi.staticfiles import StaticFiles
import io 
from collections import Counter

class serverApi:
    def __init__(self,host:str,port:int) -> None:
        self.port = port
        self.host = host
        self.model = YOLO("./model/yolov8m.pt") 
        self.api = FastAPI(title="Self Checkout Api")
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.api.add_api_route("/status",self.serve_status,methods=['GET'])
        self.api.add_api_route("/inference",self.inference,methods=["POST"])
        self.api.mount("/", StaticFiles(directory="ui", html=True), name="ui")
    async def serve_status(self):
        return {
            "status":"alive"
        }
    
    async def inference(self,file:UploadFile=File(...)):
        image_binary = await file.read()
        img_to_model = Image.open(io.BytesIO(image_binary))
        results = self.model(img_to_model)
        names = self.model.names
        list_of_items = []
        for r in results:
            for c in r.boxes.cls:
                list_of_items.append(names[int(c)])
        print(list_of_items)
        result = Counter(list_of_items)
        content = dict(result)
        return {"content":content}
    
    def start_server(self):
        config = uvicorn.Config(self.api,self.host,self.port)
        server = uvicorn.Server(config)
        server.run()





if __name__ == "__main__":
    my_server = serverApi("localhost",8000)
    my_server.start_server()

