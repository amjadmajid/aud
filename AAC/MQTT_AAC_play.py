#!/usr/bin/env python

import paho.mqtt.client as mqtt

try:
    from pixel_ring import pixel_ring
    from gpiozero import LED
    on_pi = True
except ImportError:
    on_pi = False

import sounddevice as sd
from scipy.io.wavfile import read
import time
from generic_audio_functions import play_file


def playing(args):
    args = args.split()

    test = '-t' in args
    
    music_name = args[args.index('--music')+1]
    music_file = '../AAC/sample_chirps/{}.wav'.format(music_name)
    
    duration = float(args[args.index('--duration')+1])
    padding = float(args[args.index('--padding')+1])

    if on_pi:
        power = LED(5)
        power.on()
        pixel_ring.set_brightness(100)
        pixel_ring.listen()
        # pixel_ring.think()

    if test:
        #test mode
        time.sleep(duration)
    else:
        play_file(music_file, add_padding_to_end=True, padding_duration_ms=padding*1000)

    if on_pi:
        pixel_ring.off()
        power.off()


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    client.subscribe("playrec")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    msg.payload = msg.payload.decode('utf-8')
    print(msg.topic+" "+str(msg.payload))

    playing(msg.payload)
    #Used for testing
    #print "lightshow"
    #subsystem.call(['python', 'Lightshow2.py'])
    print ('done')
    client.publish("play_done",1)

client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

#connect
print("Connecting...")
client.connect("192.168.1.196")

print("Looping...")
client.loop_start()
while True:
    try:
        time.sleep(0.1)
    except KeyboardInterrupt:
        break

client.loop_stop()
