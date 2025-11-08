#!/usr/bin/env python3
"""
Helper script to check microphone permissions and list available devices
"""

import sys

print("🔍 Checking microphone setup...\n")

# Check pyaudio
try:
    import pyaudio
    print("✅ pyaudio is installed")
    p = pyaudio.PyAudio()
    
    print("\n📱 Available audio input devices:")
    print("-" * 60)
    default_input = p.get_default_input_device_info()
    print(f"Default Input Device: {default_input['name']} (Index: {default_input['index']})")
    print(f"  Channels: {default_input['maxInputChannels']}")
    print(f"  Sample Rate: {default_input['defaultSampleRate']} Hz")
    print()
    
    print("All Input Devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            marker = " ← DEFAULT" if i == default_input['index'] else ""
            print(f"  [{i}] {info['name']}{marker}")
            print(f"      Channels: {info['maxInputChannels']}, "
                  f"Sample Rate: {info['defaultSampleRate']} Hz")
    
    print("\n🎤 Testing microphone access...")
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        # Try to read a chunk
        data = stream.read(1024, exception_on_overflow=False)
        
        # Calculate level
        import struct
        samples = struct.unpack('1024h', data)
        rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
        level = rms / 32768.0
        
        stream.stop_stream()
        stream.close()
        
        if level < 0.001:
            print("⚠️  Microphone is accessible but appears silent (0.0% level)")
            print("   This could mean:")
            print("   - Microphone is muted")
            print("   - No sound is being captured")
            print("   - Wrong device selected")
        else:
            print(f"✅ Microphone is working! Audio level: {level*100:.1f}%")
            
    except OSError as e:
        if "Input overflowed" in str(e):
            print("⚠️  Input overflow (this is usually OK)")
        else:
            print(f"❌ Error accessing microphone: {e}")
            print("\n💡 MICROPHONE PERMISSION REQUIRED!")
            print("   Follow these steps:")
            print("   1. Open System Settings (or System Preferences)")
            print("   2. Go to Privacy & Security")
            print("   3. Click Microphone")
            print("   4. Enable Terminal (or Python if it appears)")
            print("   5. Restart this script")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if "Permission denied" in str(e) or "access denied" in str(e).lower():
            print("\n💡 MICROPHONE PERMISSION REQUIRED!")
            print("   Follow these steps:")
            print("   1. Open System Settings (or System Preferences)")
            print("   2. Go to Privacy & Security")
            print("   3. Click Microphone")
            print("   4. Enable Terminal (or Python if it appears)")
            print("   5. Restart this script")
            sys.exit(1)
    
    p.terminate()
    
except ImportError:
    print("❌ pyaudio is not installed")
    print("   Install with: pip install pyaudio")
    sys.exit(1)

print("\n" + "=" * 60)
print("📋 MICROPHONE PERMISSION INSTRUCTIONS FOR macOS:")
print("=" * 60)
print("""
If microphone access was denied, follow these steps:

1. Open System Settings (or System Preferences on older macOS)
   - Click the Apple menu (🍎) > System Settings
   - Or search for "System Settings" in Spotlight

2. Navigate to Privacy & Security
   - Click "Privacy & Security" in the sidebar
   - Or search for "Privacy" in the search box

3. Click Microphone
   - Scroll down to find "Microphone" in the list
   - Click on it

4. Enable Terminal (or Python)
   - Look for "Terminal" in the list
   - Toggle the switch to ON (green)
   - If you see "Python" instead, enable that

5. Restart the script
   - Close this terminal window
   - Open a new terminal
   - Run the script again

Alternative: If Terminal doesn't appear in the list:
- Run the script once (it will fail)
- macOS should show a permission prompt
- Click "OK" to grant permission
- Or manually add Terminal in System Settings

Troubleshooting:
- If still not working, try restarting your Mac
- Check that your microphone is not muted in System Settings > Sound
- Try a different microphone if available
""")


