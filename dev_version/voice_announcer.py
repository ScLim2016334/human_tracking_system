"""
Voice Announcer Module
Handles voice announcements for occlusion scenarios
"""

import time
from config import *

class VoiceAnnouncer:
    def __init__(self):
        self.last_announcement_time = 0
        self.announcement_cooldown = 2.0  # Minimum time between announcements (seconds)
        
        # Voice messages in different languages
        self.messages = {
            "en": {
                "stop_request": "Master please stop for {} seconds",
                "look_request": "Master please look at me",
                "cant_see": "Master I can't see you, please come in front of me"
            },
            "zh": {
                "stop_request": "主人请停止{}秒钟",
                "look_request": "主人请看着我",
                "cant_see": "主人我看不到您，请到我前面来"
            }
        }
    
    def can_announce(self):
        """Check if enough time has passed since last announcement"""
        current_time = time.time()
        return current_time - self.last_announcement_time >= self.announcement_cooldown
    
    def announce_stop_request(self, stop_time):
        """Announce request for master to stop"""
        if not ENABLE_VOICE_ANNOUNCEMENTS or not self.can_announce():
            return
        
        message = self.messages[VOICE_LANGUAGE]["stop_request"].format(stop_time)
        self._speak_message(message)
        print(f"🔊 Voice Announcement: {message}")
    
    def announce_look_request(self):
        """Announce request for master to look at the robot"""
        if not ENABLE_VOICE_ANNOUNCEMENTS or not self.can_announce():
            return
        
        message = self.messages[VOICE_LANGUAGE]["look_request"]
        self._speak_message(message)
        print(f"🔊 Voice Announcement: {message}")
    
    def announce_cant_see(self):
        """Announce that master cannot be seen"""
        if not ENABLE_VOICE_ANNOUNCEMENTS or not self.can_announce():
            return
        
        message = self.messages[VOICE_LANGUAGE]["cant_see"]
        self._speak_message(message)
        print(f"🔊 Voice Announcement: {message}")
    
    def _speak_message(self, message):
        """
        Actually speak the message.
        This is a placeholder - you can integrate with your preferred TTS system.
        """
        # Option 1: Using pyttsx3 (cross-platform)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
            self.last_announcement_time = time.time()
        except ImportError:
            # Option 2: Using Windows SAPI (Windows only)
            try:
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Speak(message)
                self.last_announcement_time = time.time()
            except ImportError:
                # Option 3: Using espeak (Linux)
                try:
                    import subprocess
                    subprocess.run(["espeak", message], check=True)
                    self.last_announcement_time = time.time()
                except (ImportError, FileNotFoundError, subprocess.CalledProcessError):
                    # Fallback: just print the message
                    print(f"🔊 [TTS not available] {message}")
                    self.last_announcement_time = time.time()

# Global voice announcer instance
voice_announcer = VoiceAnnouncer() 