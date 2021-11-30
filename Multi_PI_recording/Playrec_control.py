import paho.mqtt.client as mqtt
import time

def play_done_callback(client, userdata, message):
    global play_done
    play_done = True
    print("play_done")

def rec_done_callback(client, userdata, message):
    global rec_done
    rec_done = True
    print("rec_done")

def on_message(client, userdata, message):
    print(msg.topic+" "+str(msg.payload))

def Input_parsing(dist, direction, LoS, edist, edirection, location, top, test, duration):

    args = ""
    if test:
        args = "{} -t".format(args)

    args = "{} --distance {}".format(args, dist)
    args = "{} --direction {}".format(args, direction)

    if not LoS:
        args = "{} --LoS n".format(args)
        args = "{} --edistance n".format(args, edist)
        args = "{} --edirectio n".format(args, edirection)
        
    args = "{} --top {}".format(args, top)
    args = "{} --location {}".format(args, location)

    args = "{} --duration {}".format(args, duration)

    return args



# User inputs start

# (Geodesic) souce location
dist = 50 #cm
direction = 0

# Line-of-Sight state
LoS = True

# (Euclidian) source location (only send if LoS == False)
edist = 0 #cm
edirection = 0

# Meta lobation of recording
location = 0

# Top state (if ther's an obustructio inbetween the mics)
top = "y"

# Testing flag (set to True to run trough program without actually playing/recording
test = False

# Music_files
# File names
num_chirp_trains = 2

music_names = []
for i in range(num_chirp_trains):
    music_names.append('chirp_train_chirp{}'.format(i))

# Length of the music files (seconds)
duration = 2#5

# User inputs end
msg = Input_parsing(dist, direction, LoS, edist, edirection, location, top, test, duration)




rec_init = True
play_init = False

rec_done = rec_init
play_done = play_init
    



client = mqtt.Client()

client.connect("Robomindpi-002")

client.subscribe("rec_done")
client.subscribe("play_done")

client.message_callback_add("rec_done", rec_done_callback)
client.message_callback_add("play_done", play_done_callback)

client.loop_start()

print("playrec settings \n{}".format(msg))
for i in range(num_chirp_trains):
    print(music_names[i])
    
    rec_done = rec_init
    play_done = play_init
    
    tx_args = "{} --music {}".format(msg,music_names[i])
    print(tx_args)
    client.publish("playrec", tx_args)

    while not (rec_done and play_done):
        pass
    
    time.sleep(1)

print("done")
client.loop_stop()
