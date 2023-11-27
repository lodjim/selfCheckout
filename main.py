from fastapi import FastAPI,File, UploadFile,Response
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from fastapi.staticfiles import StaticFiles
import io 
import cv2
import asyncio
from fastapi.responses import StreamingResponse
from collections import Counter

class serverApi:
    def __init__(self,host:str,port:int) -> None:
        self.port = port
        self.host = host
        self.model = YOLO("./model/yolov8m.pt") 
        self.api = FastAPI(title="Self Checkout Api")
        self.camera_lock = asyncio.Lock()
        self.camera = cv2.VideoCapture(0)
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.api.add_api_route("/status",self.serve_status,methods=['GET'])
        self.api.add_api_route("/inference",self.inference,methods=["GET"])
        self.api.add_api_route("/get-flux",self.video_feed,methods=['GET'])

        self.api.mount("/", StaticFiles(directory="ui", html=True), name="ui")
    async def serve_status(self):
        return {
            "status":"alive"
        }
    
    async def get_frame(self):
        async with self.camera_lock:
            success, frame = await asyncio.to_thread(self.camera.read)
            if success:
                return frame
            else:
                return None  # Or handle the error appropriately
            
    async def generate_frames(self):
        while True:
            frame = await self.get_frame()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.01)

   
    async def video_feed(self):
        return StreamingResponse(self.generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

    async def inference(self):
        print("receving.......")
        _,image_binary = await asyncio.to_thread(self.camera.read)
        results = self.model(image_binary)
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
    my_server = serverApi("0.0.0.0",8000)
    my_server.start_server()

