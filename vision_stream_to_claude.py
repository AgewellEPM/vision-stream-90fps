#!/usr/bin/env python3
"""
90 FPS Vision Streaming to Claude
Captures screen at 90 FPS, parses content, and streams to Claude
"""

import cv2
import numpy as np
import time
import threading
import queue
import mss
import pytesseract
from datetime import datetime
import json
import os

class VisionStreamToClaude:
    def __init__(self, fps=90):
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.running = False
        self.frame_queue = queue.Queue(maxsize=100)
        self.parsed_queue = queue.Queue(maxsize=100)
        
        # Screen capture
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[0]  # Primary monitor
        
        # Stats
        self.frames_captured = 0
        self.frames_parsed = 0
        self.start_time = None
        
    def capture_frames(self):
        """Capture frames at 90 FPS"""
        print(f"📸 Starting 90 FPS screen capture...")
        
        while self.running:
            frame_start = time.time()
            
            # Capture screen
            screenshot = self.sct.grab(self.monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Add to queue if not full
            if not self.frame_queue.full():
                self.frame_queue.put({
                    'frame': frame,
                    'timestamp': datetime.now().isoformat(),
                    'frame_num': self.frames_captured
                })
                self.frames_captured += 1
            
            # Maintain FPS
            elapsed = time.time() - frame_start
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def parse_frames(self):
        """Parse frames and extract information"""
        print("🧠 Starting frame parser...")
        
        while self.running or not self.frame_queue.empty():
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                frame = frame_data['frame']
                
                # Parse frame content
                parsed = {
                    'timestamp': frame_data['timestamp'],
                    'frame_num': frame_data['frame_num'],
                    'data': {}
                }
                
                # 1. Detect significant changes
                if hasattr(self, 'last_frame'):
                    diff = cv2.absdiff(self.last_frame, frame)
                    change_score = np.mean(diff)
                    parsed['data']['change_score'] = float(change_score)
                    
                    if change_score > 10:  # Significant change
                        parsed['data']['significant_change'] = True
                
                # 2. Extract text (OCR)
                if frame_data['frame_num'] % 30 == 0:  # Every 30 frames
                    try:
                        text = pytesseract.image_to_string(frame)
                        if text.strip():
                            parsed['data']['text'] = text.strip()[:500]  # Limit text
                    except:
                        pass
                
                # 3. Detect dominant colors
                if frame_data['frame_num'] % 45 == 0:  # Every 45 frames
                    avg_color = frame.mean(axis=0).mean(axis=0)
                    parsed['data']['dominant_color'] = {
                        'r': int(avg_color[2]),
                        'g': int(avg_color[1]),
                        'b': int(avg_color[0])
                    }
                
                # 4. Motion detection
                if hasattr(self, 'last_frame'):
                    flow = cv2.calcOpticalFlowFarneback(
                        cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                    if motion_magnitude > 2:
                        parsed['data']['motion'] = {
                            'magnitude': float(motion_magnitude),
                            'detected': True
                        }
                
                self.last_frame = frame.copy()
                self.frames_parsed += 1
                
                # Add to parsed queue
                if parsed['data']:  # Only if we have data
                    self.parsed_queue.put(parsed)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Parse error: {e}")
                
    def stream_to_claude(self):
        """Stream parsed data to Claude-readable format"""
        print("📡 Starting Claude stream...")
        
        # Create output directory
        os.makedirs('claude_vision_stream', exist_ok=True)
        
        batch = []
        batch_size = 10
        
        while self.running or not self.parsed_queue.empty():
            try:
                parsed = self.parsed_queue.get(timeout=0.1)
                batch.append(parsed)
                
                # Send batch when full or on significant events
                if len(batch) >= batch_size or parsed['data'].get('significant_change'):
                    self.send_to_claude(batch)
                    batch = []
                    
            except queue.Empty:
                if batch:  # Send remaining
                    self.send_to_claude(batch)
                    batch = []
            except Exception as e:
                print(f"❌ Stream error: {e}")
                
    def send_to_claude(self, batch):
        """Format and send batch to Claude"""
        # Create Claude-readable summary
        summary = {
            'batch_time': datetime.now().isoformat(),
            'frame_count': len(batch),
            'events': []
        }
        
        for item in batch:
            if 'text' in item['data']:
                summary['events'].append({
                    'type': 'text_detected',
                    'content': item['data']['text'][:200],
                    'frame': item['frame_num']
                })
            
            if item['data'].get('significant_change'):
                summary['events'].append({
                    'type': 'screen_change',
                    'score': item['data']['change_score'],
                    'frame': item['frame_num']
                })
                
            if 'motion' in item['data']:
                summary['events'].append({
                    'type': 'motion_detected',
                    'magnitude': item['data']['motion']['magnitude'],
                    'frame': item['frame_num']
                })
        
        # Output to terminal (Claude can read this)
        if summary['events']:
            print("\n" + "="*50)
            print("🎯 VISION UPDATE TO CLAUDE:")
            print(f"⏱️ Time: {summary['batch_time']}")
            print(f"📊 Frames processed: {summary['frame_count']}")
            print("📌 Events:")
            for event in summary['events'][:5]:  # Limit to 5 events
                print(f"  • {event['type']}: {event}")
            print("="*50 + "\n")
            
        # Also save to file
        filename = f"claude_vision_stream/vision_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def start(self):
        """Start all threads"""
        self.running = True
        self.start_time = time.time()
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_frames)
        parse_thread = threading.Thread(target=self.parse_frames)
        stream_thread = threading.Thread(target=self.stream_to_claude)
        
        capture_thread.start()
        parse_thread.start() 
        stream_thread.start()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║           90 FPS VISION STREAMING TO CLAUDE                  ║
╚══════════════════════════════════════════════════════════════╝

✅ System Active:
  • Capturing at 90 FPS
  • Parsing frames for text, motion, changes
  • Streaming events to Claude
  
📊 Stats updating every second...
Press Ctrl+C to stop
""")
        
        # Monitor performance
        try:
            while self.running:
                time.sleep(1)
                elapsed = time.time() - self.start_time
                actual_fps = self.frames_captured / elapsed
                parse_fps = self.frames_parsed / elapsed
                
                print(f"\r📊 Capture: {actual_fps:.1f} FPS | Parse: {parse_fps:.1f} FPS | Queue: {self.frame_queue.qsize()}", end='')
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            self.running = False
            
        # Wait for threads
        capture_thread.join()
        parse_thread.join()
        stream_thread.join()
        
        print("✅ Vision stream stopped")
        
def main():
    streamer = VisionStreamToClaude(fps=90)
    streamer.start()

if __name__ == "__main__":
    main()