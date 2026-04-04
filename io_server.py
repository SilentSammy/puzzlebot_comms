from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import cv2
import json
import numpy as np
from typing import Dict, Callable, Any
import threading
import uvicorn
import time

app = FastAPI(title="Robot I/O Server")

# Endpoint registries - developers populate these
http_endpoints: Dict[str, Callable] = {}
stream_endpoints: Dict[str, Callable] = {}

port = 5000


def start_webserver(threaded=True):
    """Start the FastAPI server with uvicorn"""
    config = uvicorn.Config(
        app,
        host='0.0.0.0',
        port=port,
        log_level='info'
    )
    server = uvicorn.Server(config)
    
    if threaded:
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
    else:
        server.run()


def _is_image_data(data) -> bool:
    """Detect if data is an image (numpy array with 2 or 3 dimensions)"""
    return isinstance(data, np.ndarray) and len(data.shape) in [2, 3]


def _serialize_data(data) -> tuple[bytes, str]:
    """
    Serialize data for transmission.
    Returns: (serialized_bytes, data_type)
    data_type: 'image', 'json'
    """
    if isinstance(data, np.ndarray):
        if _is_image_data(data):
            # Encode image as JPEG
            ret, jpeg = cv2.imencode('.jpg', data)
            if ret:
                return jpeg.tobytes(), 'image'
        # Non-image arrays → JSON
        return json.dumps(data.tolist()).encode('utf-8'), 'json'
    elif isinstance(data, (dict, list, int, float, str, bool)):
        return json.dumps(data).encode('utf-8'), 'json'
    else:
        # Fallback: convert to string
        return str(data).encode('utf-8'), 'json'


# Root endpoint - shows available endpoints
@app.get("/")
async def root():
    """List all available endpoints"""
    return {
        "http_endpoints": list(http_endpoints.keys()),
        "stream_endpoints": list(stream_endpoints.keys()),
        "usage": {
            "http": "GET/POST /{endpoint_name}",
            "stream_mjpeg": "GET /stream/{name} (cameras only)",
            "stream_websocket": "WS /ws/{name} (all sensors)"
        }
    }


# MJPEG streaming for cameras - HTTP endpoint
@app.get("/stream/{name}")
async def mjpeg_stream(name: str):
    """
    Serve camera streams as MJPEG (Motion JPEG) over HTTP.
    Only works for image data sources.
    """
    callback = stream_endpoints.get(name)
    if not callback:
        return JSONResponse(content={"error": "Stream not defined"}, status_code=404)
    
    # Test if callback returns image data
    try:
        sample = callback()
        if not _is_image_data(sample):
            return JSONResponse(
                content={
                    "error": "MJPEG streaming only supports image data. Use WebSocket at /ws/{name} for this sensor."
                },
                status_code=400
            )
    except Exception as e:
        return JSONResponse(content={"error": f"Stream callback error: {e}"}, status_code=500)
    
    # Generate MJPEG stream
    async def generate():
        while True:
            try:
                frame = callback()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       jpeg.tobytes() + b'\r\n')
                
                await asyncio.sleep(0.001)  # Prevent blocking
            except Exception as e:
                print(f"Error in MJPEG stream '{name}': {e}")
                await asyncio.sleep(0.1)
    
    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


# WebSocket streaming - works for all data types
@app.websocket("/ws/{name}")
async def websocket_stream(websocket: WebSocket, name: str):
    """
    Stream data via WebSocket. Auto-detects data type:
    - Images: sent as binary JPEG
    - Other data: sent as JSON text
    """
    await websocket.accept()
    
    callback = stream_endpoints.get(name)
    if not callback:
        await websocket.send_text(json.dumps({"error": "Stream not defined"}))
        await websocket.close()
        return
    
    try:
        while True:
            try:
                data = callback()
                
                if data is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Serialize data
                serialized, data_type = _serialize_data(data)
                
                # Send based on type
                if data_type == 'image':
                    await websocket.send_bytes(serialized)
                else:
                    await websocket.send_text(serialized.decode('utf-8'))
                
                await asyncio.sleep(0.001)  # Prevent overwhelming the connection
                
            except Exception as e:
                print(f"Error in WebSocket stream '{name}': {e}")
                await asyncio.sleep(0.1)
                
    except Exception as e:
        print(f"WebSocket connection closed for '{name}': {e}")
    finally:
        await websocket.close()


# HTTP endpoints - dynamic routing (catch-all must be LAST)
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
async def catch_all_http(path: str, request: Request):
    """Route HTTP requests to registered callbacks"""
    # Debug logging
    print(f"\n=== Received Request ===")
    print(f"URL Path: {path}")
    print(f"Method: {request.method}")
    print(f"Query Params: {dict(request.query_params)}")
    
    # Try to parse body
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            print(f"JSON Data: {body}")
        elif content_type:
            body = await request.body()
            print(f"Raw Data: {body[:200]}")  # Truncate for display
    except Exception as e:
        print(f"Body parse error: {e}")
    
    # Check for registered HTTP endpoint
    callback = http_endpoints.get(path)
    if callback:
        try:
            result = callback(request)
            # Handle tuple returns (message, status_code)
            if isinstance(result, tuple) and len(result) == 2:
                return JSONResponse(content={"message": result[0]}, status_code=result[1])
            return result
        except Exception as e:
            print(f"Error in HTTP callback: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    return JSONResponse(content={"error": "Endpoint not defined"}, status_code=404)


if __name__ == '__main__':
    # Example usage/demo
    x, w = 0.0, 0.0
    
    def receive_vel(request: Request):
        """Example: HTTP endpoint for velocity commands"""
        global x, w
        params = dict(request.query_params)
        x = float(params.get('x', x))
        w = float(params.get('w', w))
        return f"Linear Velocity: {x}, Angular Velocity: {w}", 200
    
    def camera_source():
        """Example: Camera stream (returns numpy array)"""
        global default_cap
        if 'default_cap' not in globals():
            default_cap = cv2.VideoCapture(0)
        ret, frame = default_cap.read()
        return frame if ret else None
    
    # Register endpoints
    http_endpoints['cmd_vel'] = receive_vel
    stream_endpoints['camera'] = camera_source
    
    print("\n" + "="*60)
    print("🤖 Robot I/O Server Starting...")
    print("="*60)
    print(f"📡 HTTP endpoints: {list(http_endpoints.keys())}")
    print(f"📊 Stream endpoints: {list(stream_endpoints.keys())}")
    print(f"\n📸 Camera MJPEG: http://localhost:{port}/stream/camera")
    print(f"🎮 Velocity Command: http://localhost:{port}/cmd_vel?x=0.5&w=0.2")
    print("="*60 + "\n")
    
    start_webserver(threaded=False)
