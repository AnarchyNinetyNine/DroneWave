#!/usr/bin/env python3

import cmd
import asyncio
import json
from ultralytics import YOLO
from mavsdk import System

class DroneCLI(cmd.Cmd):
    intro = "Welcome to the Drone control CLI. Type 'help' or '?' to list commands."
    prompt = "(drone) "

    def __init__(self):
        super().__init__()
        self.model = YOLO('./best.pt')
        self.drone = System()
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.connect_drone())

    async def connect_drone(self):
        print("🔌 Connecting to drone...")
        await self.drone.connect(system_address="udp://:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("✅ Drone connected")
                break

        print("📡 Waiting for GPS & home lock...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("✅ Position lock acquired")
                break

        print("🔋 Drone ready for commands.")

    def do_detect(self, arg):
        "Detect objects in an image: detect <image_path>"
        image_path = arg.strip()
        if not image_path:
            print("Please provide the image path.")
            return
        
        # Run detection synchronously by running async function in the loop
        self.loop.run_until_complete(self.handle_detection(image_path))

    async def handle_detection(self, image_path):
        results = self.model(image_path)
        result = results[0]

        # Extract detections into a structured list
        detections = []
        highest_confidence = 0
        top_class = None

        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_id = int(cls.item())
            class_name = result.names[class_id]
            confidence = conf.item()
            coords = box.tolist()

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "bbox": [round(c, 2) for c in coords]
            })

            if confidence > highest_confidence:
                highest_confidence = confidence
                top_class = class_name

        # Print prettified JSON of results
        print(json.dumps({
            "image": image_path,
            "detections": detections,
            "top_detection": {
                "class": top_class,
                "confidence": round(highest_confidence, 2)
            }
        }, indent=4))

        # Trigger drone action based on top class
        if top_class:
            top_class_lower = top_class.lower()
            if "takeoff" in top_class_lower:
                print("🚀 'takeoff' detected — initiating drone takeoff sequence.")
                await self.takeoff()
            elif "land" in top_class_lower:
                print("🛬 'land' detected — initiating drone landing sequence.")
                await self.land()

    async def takeoff(self):
        print("🔋 Arming drone...")
        await self.drone.action.arm()

        print("🛫 Taking off to 10 meters...")
        await self.drone.action.set_takeoff_altitude(10)
        await self.drone.action.takeoff()
        await asyncio.sleep(10)

    async def land(self):
        print("🛬 Landing...")
        await self.drone.action.land()
        await asyncio.sleep(10)

    def do_exit(self, arg):
        "Exit the CLI"
        print("Exiting Drone CLI...")
        return True


if __name__ == "__main__":
    DroneCLI().cmdloop()

